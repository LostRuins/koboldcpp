#include "parler_model.h"

// For loading parler model from gguf file.
static const std::map<std::string, parler_tensor> PARLER_TENSOR_GGUF_LOOKUP = {
    {"layer_norm.weight", PARLER_NORM},
    {"layer_norm.bias", PARLER_NORM_BIAS},
    {"embed_prompts", PARLER_EMBD_PROMPTS},
    {"text_encoding", PARLER_TEXT_ENCODING},
    {"positional_embed", PARLER_POSITIONAL_EMBD},
    {".self_attn.q_proj.weight", PARLER_LAYER_SELF_ATTN_Q},
    {".self_attn.k_proj.weight", PARLER_LAYER_SELF_ATTN_K},
    {".self_attn.v_proj.weight", PARLER_LAYER_SELF_ATTN_V},
    {".self_attn.out_proj.weight", PARLER_LAYER_SELF_ATTN_O},
    {".self_attn_layer_norm.weight", PARLER_LAYER_SELF_ATTN_NORM},
    {".self_attn_layer_norm.bias", PARLER_LAYER_SELF_ATTN_NORM_BIAS},
    {".encoder_attn.q_proj.weight", PARLER_LAYER_ATTN_Q},
    {".encoder_attn.k_proj.weight", PARLER_LAYER_ATTN_K},
    {".encoder_attn.v_proj.weight", PARLER_LAYER_ATTN_V},
    {".encoder_attn.out_proj.weight", PARLER_LAYER_ATTN_O},
    {".encoder_attn_layer_norm.weight", PARLER_LAYER_ATTN_NORM},
    {".encoder_attn_layer_norm.bias", PARLER_LAYER_ATTN_NORM_BIAS},
    {".fc1.weight", PARLER_LAYER_FC1},
    {".fc2.weight", PARLER_LAYER_FC2},
    {".final_layer_norm.weight", PARLER_LAYER_OUT_NORM},
    {".final_layer_norm.bias", PARLER_LAYER_OUT_NORM_BIAS},
    {".weight", PARLER_EMBD},
    {".weight.head", PARLER_HEAD}
};

void parler_tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_decoder(this, name, tensor);
}

void parler_tts_model::prep_layers(gguf_context * meta_ctx) {
    layers.reserve((size_t) n_layers);
    for (int i = 0; i < (int) n_layers; i++) {
        parler_layer * l = new parler_layer{};
        layers.push_back(l);
    }

    embds.reserve((size_t) n_output_heads);
    heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * h = nullptr;
        struct ggml_tensor * embd = nullptr;
        embds.push_back(embd);
        heads.push_back(h);
    }
}

void parler_tts_model::prep_constants(gguf_context * meta) {
    int encode_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.encode_length", "encode_length"});
    if (encode_length_key == -1) {
        TTS_ABORT("key 'parler-tts.decoder.encode_length' must be specified in gguf file.");
    }
    n_encode_length = gguf_get_val_u32(meta, encode_length_key);

    int hidden_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.hidden_size", "hidden_size"});
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads"});
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }
    int ctx_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.context_length", "ctx_length"});
    if (ctx_length_key != -1) {
        max_ctx_length = gguf_get_val_u32(meta, ctx_length_key);
    }

    int attn_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.attention.head_count", "attn_heads"});
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }
    head_size = hidden_size / n_attn_heads;
    max_cross_nodes = n_attn_heads * 2;

    int output_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.out_vocab_size", "out_vocab_size"});
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }

    int audio_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.audio_vocab_size", "audio_vocab_size"});
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }

    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }

    int n_layers_key = search_for_gguf_keys(meta, {"parler-tts.decoder.num_hidden_layers", "num_hidden_layers"});
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

    int bos_token_id_key = search_for_gguf_keys(meta, {"audio.bos_token_id", "bos_token_id"});
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = search_for_gguf_keys(meta, {"audio.eos_token_id", "eos_token_id"});
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }
}

void parler_tts_model::prep_cross_key_values(int n_threads, struct tts_response * conditional_prompt) {
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    // Let it create a disposable threadpool just this once
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
    std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
    std::vector<ggml_backend_t> backs = {backend_cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_cross_nodes*n_layers, false, false);

    std::vector<uint8_t> buf_compute_meta;
    buf_compute_meta.resize(max_cross_nodes*n_layers*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_cross_nodes*n_layers, false));

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * cctx = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(cctx, 4096, false);
    if (conditional_prompt) {
        // If we are updating the conditional prompt then we have to reset the tensor offsets into the ggml_context otherwise we could overflow the assigned buffer and lose our prompt.
        // These offsets are assigned by #set_tensor below.
        offset -= n_encode_length*hidden_size*sizeof(float)*n_layers*2;
        precomputed_input_emb = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, conditional_prompt->hidden_size, conditional_prompt->n_outputs);
        ggml_set_input(precomputed_input_emb);
        n_encode_length = conditional_prompt->n_outputs;
    }

    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * Kcur = ggml_mul_mat(cctx, layers[i]->attn_k_proj, precomputed_input_emb);
        struct ggml_tensor * Vcur = ggml_mul_mat(cctx, layers[i]->attn_v_proj, precomputed_input_emb);

        Kcur = ggml_reshape_3d(cctx, Kcur, head_size, n_attn_heads, n_encode_length);
        Vcur = ggml_transpose(cctx, Vcur);

        struct ggml_tensor * k = ggml_cont(cctx, ggml_permute(cctx, Kcur, 0, 2, 1, 3));
        ggml_set_name(k, ("cross_key_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, k);

        struct ggml_tensor * v = ggml_cont_3d(cctx, Vcur, n_encode_length, head_size, n_attn_heads);
        ggml_set_name(v, ("cross_value_" + std::to_string(i)).c_str());
        ggml_build_forward_expand(gf, v);
    }

    ggml_free(cctx);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);
    if (conditional_prompt) {
        ggml_backend_tensor_set(precomputed_input_emb, conditional_prompt->data, 0, conditional_prompt->n_outputs*conditional_prompt->hidden_size*ggml_element_size(precomputed_input_emb));
    }

    ggml_backend_sched_graph_compute_async(sched, gf);

    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * k = ggml_graph_get_tensor(gf, ("cross_key_" + std::to_string(i)).c_str());
        layers[i]->cross_k = ggml_dup_tensor(ctx, k);
        set_tensor(layers[i]->cross_k, k);
        struct ggml_tensor * v = ggml_graph_get_tensor(gf, ("cross_value_" + std::to_string(i)).c_str());
        layers[i]->cross_v = ggml_dup_tensor(ctx, v);
        set_tensor(layers[i]->cross_v, v);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cpu);
}

void assign_parler_layer(parler_tts_model * model, parler_layer * layer, std::string name, ggml_tensor * tensor) {
    try {
        switch(PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
            case PARLER_LAYER_SELF_ATTN_Q:
                layer->self_attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_q_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_K:
                layer->self_attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_k_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_V:
                layer->self_attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_v_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_O:
                layer->self_attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_o_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM:
                layer->self_attn_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM_BIAS:
                layer->self_attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm_bias, tensor);
                break;
            case PARLER_LAYER_ATTN_Q:
                if (model->use_cross_attn) {
                    layer->attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_q_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_K:
                if (model->use_cross_attn) {
                    layer->attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_k_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_V:
                if (model->use_cross_attn) {
                    layer->attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_v_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_O:
                if (model->use_cross_attn) {
                    layer->attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_o_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM:
                if (model->use_cross_attn) {
                    layer->attn_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM_BIAS:
                if (model->use_cross_attn) {
                    layer->attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm_bias, tensor);
                }
                break;
            case PARLER_LAYER_FC1:
                layer->fc1 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc1, tensor);
                break;
            case PARLER_LAYER_FC2:
                layer->fc2 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc2, tensor);
                break;
            case PARLER_LAYER_OUT_NORM:
                layer->final_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm, tensor);
                break;
            case PARLER_LAYER_OUT_NORM_BIAS:
                layer->final_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range& e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor) {
    if (PARLER_TENSOR_GGUF_LOOKUP.find(name) != PARLER_TENSOR_GGUF_LOOKUP.end()) {
        try {
            switch (PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
                case PARLER_NORM:
                    model->layer_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm, tensor);
                    break;
                case PARLER_NORM_BIAS:
                    model->layer_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm_bias, tensor);
                    break;
                case PARLER_EMBD_PROMPTS:
                    model->prompt_embd = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->prompt_embd, tensor);
                    break;
                case PARLER_TEXT_ENCODING:
                    if (model->use_cross_attn) {
                        model->precomputed_input_emb = ggml_dup_tensor(model->ctx, tensor);
                        model->set_tensor(model->precomputed_input_emb, tensor);
                    }
                    break;
                case PARLER_POSITIONAL_EMBD:
                    model->precomputed_positional_embds = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->precomputed_positional_embds, tensor);
                    break;
                default:
                    fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                    break;
            }
        } catch (const std::out_of_range& e) {
            TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int layer = pair.first;
        std::string lt_name = pair.second;
        if (name.find("embed_tokens") != std::string::npos) {
            model->embds[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->embds[layer], tensor);
        } else if (name.find("lm_heads") != std::string::npos) {
            model->heads[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->heads[layer], tensor);
        } else {
            assign_parler_layer(model, model->layers[layer], lt_name, tensor);
        }
    }
}

void parler_context::reset(int32_t n_output_heads) {
    n_outputs = 0;
    prompt_end_position = 0;
    current_position = 0;
    output_size = 0;
    output_tokens.clear();
    eos_seen.clear();
    for (int i = 0; i < (int) n_output_heads; i++) {
        eos_seen.push_back(false);
    }
}

struct parler_context * build_new_parler_context(struct parler_tts_model * model, int n_threads, bool use_cpu) {
    parler_context * pctx = new parler_context(model, n_threads);
    pctx->eos_seen.reserve(model->n_output_heads);
    pctx->backend_cpu = ggml_backend_cpu_init();
    pctx->set_threads();
    pctx->build_schedule();
    pctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return pctx;
}

static bool parler_kv_cache_init(struct parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx, int32_t seq_id) {
    const int64_t n_layer = (int64_t) model->layers.size();
    cache->seq_id = seq_id;

    ggml_backend_buffer_type_t buft = nullptr;
    // this will only really support cpu or metal for the time being;
    if (pctx->backend != nullptr) {
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ (2u*model->n_layers+1)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;


    cache->k_l.reserve(n_layer);
    cache->v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor * k = ggml_new_tensor_1d(cache->ctx, cache->type_k, model->hidden_size*model->max_ctx_length);
        ggml_tensor * v = ggml_new_tensor_1d(cache->ctx, cache->type_v, model->hidden_size*model->max_ctx_length);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx, buft);
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    cache->buf = buf;

    return true;
}

struct ggml_tensor * parler_build_inp_embd(struct ggml_context * ctx, struct parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    // Parler has two embedding schemas one for the text input and one for generative audio tokens. These two schemas have effectively distinct shapes (i.e. [batch_size, sequence_length] and [batch_size, sequence_lenghth, num_codebooks] respectively).
    // This means that depending on where we are in generation we need to follow a distinct pattern
    struct ggml_tensor * input_embs;
    pctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(pctx->positions);
    if (batch.audio_generation) {
        pctx->audio_inp_tokens = ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_audio_tokens), batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        ggml_set_input(pctx->audio_inp_tokens);
        struct ggml_tensor * audio_tokens = ggml_reshape_2d(ctx, pctx->audio_inp_tokens, batch.n_audio_tokens / model->n_output_heads, model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            if (i == 0) {
                input_embs = ggml_get_rows(ctx, model->embds[i], ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads, audio_tokens->nb[1], i*sizeof(int32_t)));
            } else {
                input_embs = ggml_add(ctx, ggml_get_rows(ctx, model->embds[i], ggml_view_2d(ctx, audio_tokens, 1, batch.n_audio_tokens / model->n_output_heads, audio_tokens->nb[1], i*sizeof(int32_t))), input_embs);
            }
        }
    } else {
        pctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(pctx->inp_tokens);
        input_embs = ggml_get_rows(ctx, model->prompt_embd, pctx->inp_tokens);
    }
    return ggml_add(ctx, input_embs, ggml_get_rows(ctx, model->precomputed_positional_embds, pctx->positions));
}

struct ggml_tensor * parler_build_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // parler always uses default eps
    float eps = 0.00001;
    inputs = ggml_norm(ctx, inputs, eps);
    inputs = ggml_mul(ctx, inputs, weight);
    return ggml_add(ctx, inputs, bias);
}

void parler_build_kv_store(struct ggml_context * ctx, parler_kv_cache * kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa) {
    // this is the max context size;
    const int64_t n_ctx = 4096;

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[index], n_tokens*n_embd_gqa, ggml_row_size(kv->k_l[index]->type, n_embd_gqa)*kv_head);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor * v_cache_view = nullptr;

    v_cache_view = ggml_view_2d(ctx, kv->v_l[index], n_tokens, n_embd_gqa,
            (  n_ctx)*ggml_element_size(kv->v_l[index]),
            (kv_head)*ggml_element_size(kv->v_l[index]));

    v_cur = ggml_cont(ctx, ggml_transpose(ctx, v_cur));

    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

struct ggml_tensor * parler_build_head_outputs(struct ggml_context * ctx, parler_tts_model * model, struct ggml_tensor * cur) {
    // going to cat the heads together and then reshape them;
    // honestly ggml doesn't provide good support for stacking and discrete tensor access
    struct ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->heads[i], cur), 1);
        }
    }
    ggml_set_name(out, "final_out");
    //out = ggml_cont(ctx, ggml_transpose(ctx, out));

    int32_t sql_len = (int32_t) (ggml_nelements(out) / (model->output_vocab_size * model->n_output_heads));
    return ggml_cont_3d(ctx, out, model->output_vocab_size, sql_len, model->n_output_heads);
}

struct ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch) {
    pctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) pctx->current_position + batch.sequence_length, (int64_t) pctx->current_position + batch.sequence_length);
    ggml_set_input(pctx->attn_mask);

    return pctx->attn_mask;
}

struct ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    pctx->attn_mask_cross = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->n_encode_length, (int64_t) batch.sequence_length);
    ggml_set_input(pctx->attn_mask_cross);

    return pctx->attn_mask_cross;
}

static struct parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer) {
    struct parler_ubatch batch;
    batch.audio_generation = false;
    std::vector<uint32_t>* token_ids = new std::vector<uint32_t>;
    tokenizer->tokenize(sentence, *token_ids);
    token_ids->push_back(tokenizer->eos_token);
    batch.current_step = 0;
    batch.n_tokens = token_ids->size();
    batch.n_audio_tokens = 0;
    batch.sequence_length = batch.n_tokens; // sequence_length is equal to the number of tokens for non-audio generation
    std::vector<uint32_t>* position = new std::vector<uint32_t>;
    for (uint32_t i = 0; i < batch.sequence_length; i++) {
        position->push_back(i);
    }
    std::vector<uint32_t>* order = new std::vector<uint32_t>;
    for (int i = 0; i < batch.sequence_length; i++) {
        if (i >= batch.sequence_length - 1) {
            order->push_back(0);
        } else {
            order->push_back(i+1);
        }
    }
    batch.positions = position->data();
    batch.tokens = token_ids->data();
    return batch;
}

void parler_tts_runner::assign_weight(std::string name, ggml_tensor * tensor) {
    std::string::size_type pos = name.find(".", 0);
    std::string top_level(name.substr(0, pos));
    std::string value(name.substr(pos + 1));
    if (tensor->data == NULL) {
        return;
    }
    if (top_level == "audio_encoder") {
        dac_runner->model->assign_weight(value, tensor);
    } else if (top_level == "decoder") {
        model->assign_weight(value, tensor);
    } else {
        return;
    }
}

void parler_tts_runner::update_conditional_prompt(const std::string file_path, const std::string prompt, int n_threads, bool cpu_only) {
    t5_runner * text_encoder = text_encoder_from_file(file_path, n_threads, tokenizer, cpu_only);
    tts_response* response;
    text_encoder->generate(prompt, response);
    model->prep_cross_key_values(n_threads, response);
    delete text_encoder;
    return;
}


struct ggml_cgraph * parler_tts_runner::build_parler_graph(parler_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    const int32_t full_sequence_length = pctx->current_position + (uint32_t) batch.sequence_length;

    inpL = parler_build_inp_embd(ctx, pctx, model, batch);

    struct ggml_tensor * KQ_mask_dec = build_attn_mask(ctx, pctx, batch);
    struct ggml_tensor * KQ_mask_cross = build_attn_mask_cross(ctx, pctx, model, batch);

    for (int l = 0; l < model->n_layers; l++) {
        struct ggml_tensor * residual = inpL;
        ggml_set_name(inpL, ("layer_" + std::to_string(l) + "_input").c_str());

        cur = parler_build_layer_norm(ctx, inpL, model->layers[l]->self_attn_norm, model->layers[l]->self_attn_norm_bias);

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_q_proj, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_k_proj, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_v_proj, cur);

            parler_build_kv_store(ctx, kv_self, gf, Kcur, Vcur, (int32_t) batch.sequence_length, pctx->current_position, l, model->hidden_size);
            struct ggml_tensor * k =
                ggml_view_3d(ctx, kv_self->k_l[l],
                        model->head_size, full_sequence_length, model->n_attn_heads,
                        ggml_row_size(kv_self->k_l[l]->type, model->hidden_size),
                        ggml_row_size(kv_self->k_l[l]->type, model->head_size),
                        0);


            struct ggml_tensor * v =
                ggml_view_3d(ctx, kv_self->v_l[l],
                        full_sequence_length, model->head_size, model->n_attn_heads,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length*model->head_size,
                        0);

            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.sequence_length);
            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            struct ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f/sqrtf(model->head_size), 0.0f);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            attn_out = ggml_mul_mat(ctx, model->layers[l]->self_attn_o_proj, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);

        if (model->use_cross_attn) {
            struct ggml_tensor * residuala = cur;

            // norm
            cur = parler_build_layer_norm(ctx, cur, model->layers[l]->attn_norm, model->layers[l]->attn_norm_bias);

            //cross-attention
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->attn_q_proj, cur);
            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.sequence_length);

            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx, model->layers[l]->cross_k, q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_cross, 1.0f/sqrtf(model->head_size), 0.0f);

            struct ggml_tensor * kqv  = ggml_mul_mat(ctx, kq, model->layers[l]->cross_v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            cur = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.sequence_length);
            cur = ggml_mul_mat(ctx, model->layers[l]->attn_o_proj, cur);
            cur = ggml_add(ctx, cur, residuala);
        }

        struct ggml_tensor * residualffn = cur;

        cur = parler_build_layer_norm(ctx, cur, model->layers[l]->final_norm, model->layers[l]->final_norm_bias);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc1, cur);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, model->layers[l]->fc2, cur);
        cur = ggml_add(ctx, cur, residualffn);
        inpL = cur;
    }

    cur = parler_build_layer_norm(ctx, cur, model->layer_norm, model->layer_norm_bias);
    cur = parler_build_head_outputs(ctx, model, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}

void parler_tts_runner::configure_generation(generation_configuration * config) {
    sampler->temperature = config->temperature;
    sampler->repetition_penalty = config->repetition_penalty;
    sampler->do_sample = config->sample;
    sampler->top_k = config->top_k;
    sampler->top_p = config->top_p;
    model->use_cross_attn = config->use_cross_attn;
}

void parler_tts_runner::set_inputs(parler_ubatch & batch) {
    if (batch.audio_generation) {
        ggml_backend_tensor_set(pctx->audio_inp_tokens, batch.audio_tokens, 0, batch.n_audio_tokens*ggml_element_size(pctx->audio_inp_tokens));
    } else {
        ggml_backend_tensor_set(pctx->inp_tokens, batch.tokens, 0, batch.n_tokens*ggml_element_size(pctx->inp_tokens));
    }
    ggml_backend_tensor_set(pctx->positions, batch.positions, 0, batch.sequence_length*ggml_element_size(pctx->positions));
    float * d = nullptr;
    d = (float *) pctx->attn_mask->data;
    uint32_t max_pos = pctx->current_position + batch.sequence_length;
    for (int i = 0; i < batch.sequence_length; i++) {
        uint32_t pos = batch.positions[i];
        for (int ii = 0; ii < max_pos; ii++) {
            d[i*max_pos + ii] = ii > pos ? -INFINITY : 0.0f;
        }
    }

    if (model->use_cross_attn) {
        float * d2 = nullptr;
        d2 = (float *) pctx->attn_mask_cross->data;
        for (int i = 0; i < model->n_encode_length; i++) {
            for (int ii = 0; ii < batch.sequence_length; ii++) {
                d2[i*batch.sequence_length + ii] = 0.0f;
            }
        }
    }

}
void parler_tts_runner::parler_graph_compute(ggml_cgraph * gf) {
    ggml_backend_sched_graph_compute_async(pctx->sched, gf);
}

int parler_tts_runner::decode(parler_ubatch & batch) {
    ggml_backend_sched_reset(pctx->sched);

    pctx->output_tokens.reserve(model->max_generation_size);

    const size_t logits_size = model->output_vocab_size*model->max_generation_size*model->n_output_heads;
    const size_t prev_size = pctx->buf_output ? ggml_backend_buffer_get_size(pctx->buf_output) : 0;
    const size_t new_size  = logits_size * sizeof(float);

    if (!pctx->buf_output || prev_size < new_size) {
        if (pctx->buf_output) {
            ggml_backend_buffer_free(pctx->buf_output);
            pctx->buf_output = nullptr;
            pctx->logits = nullptr;
        }

        pctx->buf_output = ggml_backend_buft_alloc_buffer(pctx->backend_cpu_buffer, new_size);
    }

    pctx->logits = (float *) ggml_backend_buffer_get_base(pctx->buf_output);
    //ggml_backend_buffer_clear(pctx->buf_output, 0);

    ggml_cgraph * gf = build_parler_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(pctx->sched, gf);

    // use the sequence_length variable here so that audio input tokens are handled correctly.
    size_t n_outputs_new = batch.sequence_length;

    set_inputs(batch);
    parler_graph_compute(gf);

    float * logits_out = pctx->logits + pctx->n_outputs * model->output_vocab_size * model->n_output_heads;
    pctx->get_ggml_node_data(res, logits_out, n_outputs_new*model->output_vocab_size*model->n_output_heads*sizeof(float));

    // set to total number of outputs in the batch*/
    pctx->n_outputs += n_outputs_new;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(pctx->sched);

    return 0;
}

parler_ubatch parler_tts_runner::build_worst_case_batch()  {
    struct parler_ubatch batch;
    batch.audio_generation = false;
    batch.n_tokens = model->max_ctx_length;
    batch.n_audio_tokens = 0;
    batch.sequence_length = model->max_ctx_length;
    return batch;
}

void parler_tts_runner::prepare_post_load() {
    dac_runner->prepare_post_load();
    parler_kv_cache_init(kv_self, model, pctx, std::mt19937(std::random_device{}())());
    auto batch = build_worst_case_batch();
    auto gf = build_parler_graph(batch);
    pctx->prep_schedule(gf);
}

bool parler_tts_runner::adjust_for_sequence_continuation(struct parler_ubatch & batch) {
    return false; // not implemneted
}

bool parler_tts_runner::check_stopping() {
    int32_t token_position = (int32_t) pctx->output_tokens.size() - (int32_t) model->n_output_heads;
    if (token_position < 0) {
        return false;
    }
    if (pctx->current_position >= model->max_generation_size) {
        return true;
    }

    bool channels_complete = true;
    for (int i = 0; i < model->n_output_heads; i++) {
        pctx->eos_seen[i] = pctx->eos_seen[i] || pctx->output_tokens[token_position+i] == model->eos_token_id;
        if (channels_complete) {
            channels_complete = pctx->eos_seen[i];
        }
    }
    return channels_complete;
}

void parler_tts_runner::adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < size / model->n_output_heads; i++) {
        bool remove = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                remove = true;
                break;
            }
        }
        if (!remove) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
                if (next_index > size) {
                    filtered.push_back(model->eos_token_id);
                } else {
                    filtered.push_back(output_tokens[next_index]);
                }
            }
        }
    }
}

int parler_tts_runner::generate_from_batch(parler_ubatch & batch, struct tts_response * output) {
    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        }
        if (batch.audio_generation) {
            sampler->sample(pctx->logits + pctx->current_position * model->n_output_heads * model->output_vocab_size, pctx->output_tokens);
        }
        pctx->current_position += batch.sequence_length;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(batch.current_step > i ? pctx->eos_seen[i] ? model->eos_token_id : last_outputs[i] : model->bos_token_id);
        }
        batch = parler_ubatch{
            true, 0, 9, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step+1
        };
    }

    std::vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(pctx->output_tokens, filtered_output_tokens);
    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads, output);
    return 0;
}

int parler_tts_runner::generate_audio_tokens(std::string sentence) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    int32_t seq_id = std::mt19937(std::random_device{}())();
    if (!kv_self) {
        kv_self = new parler_kv_cache;
        if (!parler_kv_cache_init(kv_self, model, pctx, seq_id)) {
            return 1;
        }
    }

    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        }
        if (batch.audio_generation) {
            sampler->sample(pctx->logits + pctx->current_position * model->n_output_heads * model->output_vocab_size, pctx->output_tokens);
        }
        pctx->current_position += batch.sequence_length;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(batch.current_step > i ? pctx->eos_seen[i] ? model->eos_token_id : last_outputs[i] : model->bos_token_id);
        }
        batch = parler_ubatch{
            true, 0, 9, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step+1
        };
    }

    return 0;
}

void parler_tts_runner::just_audio_token_decode(uint32_t * tokens, int32_t sq_len, struct tts_response * outputs) {
    dac_runner->run(tokens, sq_len, outputs);
}

int parler_tts_runner::generate(std::string sentence, struct tts_response * output, int32_t seq_id) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    if (pctx->seq_id != seq_id || seq_id == -1) {
        seq_id = std::mt19937(std::random_device{}())();
        pctx->current_position = 0;
        if (!kv_self) {
            kv_self = new parler_kv_cache;
            if (!parler_kv_cache_init(kv_self, model, pctx, seq_id)) {
                return 1;
            }
        }
    } else {
        if (!adjust_for_sequence_continuation(batch)) {
            return 2;
        }
    }
    return generate_from_batch(batch, output);
}

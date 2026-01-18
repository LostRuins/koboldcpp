#include "dia_model.h"

void dia_model::assign_weight(std::string name, struct ggml_tensor * tensor) {
    std::vector<std::string> parts = split(name, ".");
    // GGML_LOG("\nassign_weight %s\n",name.c_str());
    if(name=="GGUF tensor data binary blob") //patch for gguf, idk why this tensor exists
    {
       return;
    }
    TTS_ASSERT(parts.size() >= 3);

    if (parts[1] == "encoder") {
        assign_to_encoder(parts, tensor, name);
    } else if (parts[1] == "decoder"){
        assign_to_decoder(parts, tensor, name);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}

void dia_model::assign_to_encoder(std::vector<std::string> parts, struct ggml_tensor * tensor, std::string name) {
    if (parts[2] == "embedding") {
        encoder->embedding = ggml_dup_tensor(ctx, tensor);
        set_tensor(encoder->embedding, tensor);
    } else if (parts[2] == "norm") {
        encoder->norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(encoder->norm, tensor);
    } else if (parts[2] == "layers") {
        TTS_ASSERT(parts.size() >= 4);
        int index = std::stoi(parts[3]);
        TTS_ASSERT(index < decoder->layers.size());
        assign_to_encoder_layer(parts[4], encoder->layers[index], tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}

void dia_model::assign_to_decoder(std::vector<std::string> parts, struct ggml_tensor * tensor, std::string name) {
    if (parts[2] == "embeddings") {
        TTS_ASSERT(parts.size() > 2);
        int index = std::stoi(parts[3]);
        TTS_ASSERT(index < decoder->embds.size());
        decoder->embds[index] = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->embds[index], tensor);
    } else if (parts[2] == "norm") {
        decoder->norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->norm, tensor);
    } else if (parts[2] == "heads") {
        TTS_ASSERT(parts.size() > 2);
        int index = std::stoi(parts[3]);
        TTS_ASSERT(index < decoder->heads.size());
        decoder->heads[index] = ggml_dup_tensor(ctx, tensor);
        set_tensor(decoder->heads[index], tensor);
    } else if (parts[2] == "layers") {
        TTS_ASSERT(parts.size() >= 4);
        int index = std::stoi(parts[3]);
        TTS_ASSERT(index < decoder->layers.size());
        assign_to_decoder_layer(parts[4], decoder->layers[index], tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' when loading Dia from GGUF file.", name.c_str());
    }
}

void dia_model::assign_to_encoder_layer(std::string part, dia_encoder_layer * layer, struct ggml_tensor * tensor) {
    if (part == "q_proj") {
        layer->q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->q, tensor);
    } else if (part == "k_proj") {
        layer->k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->k, tensor);
    } else if (part == "v_proj") {
        layer->v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->v, tensor);
    } else if (part == "o_proj") {
        layer->o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->o, tensor);
    } else if (part == "pre_sa_norm") {
        layer->self_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_norm, tensor);
    } else if (part == "post_sa_norm") {
        layer->mlp_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->mlp_norm, tensor);
    } else if (part == "gate") {
        layer->gate = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->gate, tensor);
    } else if (part == "up") {
        layer->up = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->up, tensor);
    } else if (part == "wo") {
        layer->out = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->out, tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' for encoder layer when loading Dia from GGUF file.", part.c_str());
    }
}

void dia_model::assign_to_decoder_layer(std::string part, dia_decoder_layer * layer, struct ggml_tensor * tensor) {
    if (part == "self_q_proj") {
        layer->self_attn_q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_q, tensor);
    } else if (part == "self_k_proj") {
        layer->self_attn_k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_k, tensor);
    } else if (part == "self_v_proj") {
        layer->self_attn_v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_v, tensor);
    } else if (part == "self_o_proj") {
        layer->self_attn_o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_o, tensor);
    } else if (part == "cross_q_proj") {
        layer->cross_attn_q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_q, tensor);
    } else if (part == "cross_k_proj") {
        layer->cross_attn_k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_k, tensor);
    } else if (part == "cross_v_proj") {
        layer->cross_attn_v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_v, tensor);
    } else if (part == "cross_o_proj") {
        layer->cross_attn_o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_o, tensor);
    } else if (part == "pre_sa_norm") {
        layer->self_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->self_attn_norm, tensor);
    } else if (part == "pre_mlp_norm") {
        layer->mlp_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->mlp_norm, tensor);
    } else if (part == "pre_ca_norm") {
        layer->cross_attn_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->cross_attn_norm, tensor);
    } else if (part == "gate") {
        layer->gate = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->gate, tensor);
    } else if (part == "up") {
        layer->up = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->up, tensor);
    } else if (part == "wo") {
        layer->out = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer->out, tensor);
    } else {
        TTS_ABORT("Unrecognized tensor '%s' for encoder layer when loading Dia from GGUF file.", part.c_str());
    }
}

void dia_model::prep_layers() {
    encoder = new dia_encoder;
    decoder = new dia_decoder;
    encoder->layers.reserve((size_t) n_encoder_layers);
    for (int i = 0; i < (int) n_encoder_layers; i++) {
        dia_encoder_layer * l = new dia_encoder_layer;
        encoder->layers.push_back(l);
    }

    decoder->layers.reserve((size_t) n_decoder_layers);
    for (int i = 0; i < (int) n_decoder_layers; i++) {
        dia_decoder_layer * l = new dia_decoder_layer;
        decoder->layers.push_back(l);
    }

    decoder->embds.reserve((size_t) n_output_heads);
    decoder->heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * h = nullptr;
        struct ggml_tensor * embd = nullptr;
        decoder->embds.push_back(embd);
        decoder->heads.push_back(h);
    }
}

void dia_model::prep_constants(gguf_context * meta) {
    int output_heads_key = gguf_find_key(meta, "dia.decoder.output_heads");
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }

    int decoder_layers_key = gguf_find_key(meta, "dia.decoder.layers");
    if (decoder_layers_key != -1) {
        n_decoder_layers = gguf_get_val_u32(meta, decoder_layers_key);
    }

    int encoder_layers_key = gguf_find_key(meta, "dia.encoder.layers");
    if (encoder_layers_key != -1) {
        n_encoder_layers = gguf_get_val_u32(meta, encoder_layers_key);
    }

    int decoder_hidden_size_key = gguf_find_key(meta, "dia.decoder.hidden_size");
    if (decoder_hidden_size_key != -1) {
        decoder_hidden_size = gguf_get_val_u32(meta, decoder_hidden_size_key);
    }

    int decoder_attn_heads_key = gguf_find_key(meta, "dia.decoder.attn_heads");
    if (decoder_attn_heads_key != -1) {
        decoder_attn_heads = gguf_get_val_u32(meta, decoder_attn_heads_key);
    }

    int decoder_query_heads_key = gguf_find_key(meta, "dia.decoder.query_heads");
    if (decoder_query_heads_key != -1) {
        decoder_query_heads = gguf_get_val_u32(meta, decoder_query_heads_key);
    }

    int encoder_attn_heads_key = gguf_find_key(meta, "dia.encoder.attn_heads");
    if (encoder_attn_heads_key != -1) {
        encoder_attn_heads = gguf_get_val_u32(meta, encoder_attn_heads_key);
    }

    int head_size_key = gguf_find_key(meta, "dia.attn_head_size");
    if (head_size_key != -1) {
        head_size = gguf_get_val_u32(meta, head_size_key);
    }

    int eos_token_id_key = gguf_find_key(meta, "dia.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "dia.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int pad_token_id_key = gguf_find_key(meta, "dia.pad_token_id");
    if (pad_token_id_key != -1) {
        pad_token_id = gguf_get_val_u32(meta, pad_token_id_key);
    }

    int max_context_key = gguf_find_key(meta, "dia.encoder.max_context_length");
    if (max_context_key != -1) {
        max_encoder_context_length = gguf_get_val_u32(meta, max_context_key);
    }

    int output_vocab_size_key = gguf_find_key(meta, "dia.decoder.output_vocab_size");
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }

    int audio_vocab_size_key = gguf_find_key(meta, "dia.decoder.audio_vocab_size");
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }

    int max_generation_size_key = gguf_find_key(meta, "dia.decoder.max_generation_size");
    if (max_generation_size_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_generation_size_key);
    }
    int max_delay_key = gguf_find_key(meta, "dia.max_delay");
    if (max_delay_key != -1) {
        max_delay = gguf_get_val_u32(meta, max_delay_key);
    }

    // please note that this value is not currently set in the gguf encoder as it effectively only exists as a default
    // python parameter (rather than an attribute in the model config) for the python Dia model.
    int cfg_scale_key = gguf_find_key(meta, "dia.cfg_scale");
    if (cfg_scale_key != -1) {
        cfg_scale_data[0] = gguf_get_val_f32(meta, cfg_scale_key);
    }
}

void dia_context::reset() {
    current_position = 0;
    prompt_size = 0;
    output_tokens.clear();
    delay_steps = -1;
}

struct dia_context * build_new_dia_context(struct dia_model * model, int n_threads, bool use_cpu) {
    dia_context * dctx = new dia_context(model, n_threads);
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule();
    dctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}

static bool dia_kv_cache_init(struct dia_kv_cache * cache, dia_model * model, dia_context * dctx) {
    ggml_backend_buffer_type_t buft = nullptr;
    // this will only really support cpu or metal for the time being;
    if (dctx->backend != nullptr) {

    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ (4u * model->n_decoder_layers + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;

    cache->k_l.reserve(model->n_decoder_layers);
    cache->v_l.reserve(model->n_decoder_layers);
    cache->cross_k_l.reserve(model->n_decoder_layers);
    cache->cross_v_l.reserve(model->n_decoder_layers);

    for (int i = 0; i < (int) model->n_decoder_layers; i++) {
        struct ggml_tensor * k = ggml_new_tensor_1d(cache->ctx, cache->tensor_type, model->head_size * model->decoder_attn_heads * model->max_generation_size * 2);
        struct ggml_tensor * v = ggml_new_tensor_1d(cache->ctx, cache->tensor_type, model->head_size * model->decoder_attn_heads * model->max_generation_size * 2);
        struct ggml_tensor * cross_k = ggml_new_tensor_1d(cache->ctx, cache->tensor_type, model->head_size * model->decoder_attn_heads * model->max_encoder_context_length * 2);
        struct ggml_tensor * cross_v = ggml_new_tensor_1d(cache->ctx, cache->tensor_type, model->head_size * model->decoder_attn_heads * model->max_encoder_context_length * 2);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        ggml_format_name(cross_k, "cache_cross_k_l%d", i);
        ggml_format_name(cross_v, "cache_cross_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
        cache->cross_k_l.push_back(cross_k);
        cache->cross_v_l.push_back(cross_v);
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

static struct ggml_tensor * build_dia_decoder_inp_embd(struct ggml_context * ctx, dia_context *dctx, dia_decoder * decoder, dia_ubatch & batch, uint32_t n_output_heads) {
    struct ggml_tensor * input_embs;

    dctx->audio_inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_output_heads * 2);
    ggml_set_input(dctx->audio_inp_tokens);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * view = ggml_view_1d(ctx, dctx->audio_inp_tokens, 2, i * ggml_element_size(dctx->audio_inp_tokens));
        view->nb[0] = n_output_heads * ggml_element_size(dctx->audio_inp_tokens);
        if (i == 0) {
            input_embs = ggml_get_rows(ctx, decoder->embds[i], view);
        } else {
            input_embs = ggml_add(ctx, ggml_get_rows(ctx, decoder->embds[i], view), input_embs);
        }
    }
    return input_embs;
}

static struct ggml_tensor * dia_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight) {
    // dia always uses 1e-5 as the default eps
    float eps = 0.00001;
    inputs = ggml_rms_norm(ctx, inputs, eps);
    return ggml_mul(ctx, inputs, weight);
}

static struct ggml_tensor * build_dia_encoder_attn_mask(ggml_context * ctx, struct dia_context * dctx, dia_model * model) {
    dctx->encode_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->max_encoder_context_length, (int64_t) model->max_encoder_context_length);
    ggml_set_input(dctx->encode_attn_mask);

    return dctx->encode_attn_mask;
}

static struct ggml_tensor * build_dia_head_outputs(struct ggml_context * ctx, dia_model * model, struct ggml_tensor * cur) {
    // going to cat the heads together and then reshape them
    struct ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->decoder->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->decoder->heads[i], cur), 2);
        }
    }
    struct ggml_tensor * cond = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], 0));
    struct ggml_tensor * uncond = ggml_cont(ctx, ggml_view_2d(ctx, out, out->ne[0], out->ne[2], out->nb[2], out->nb[1]));
    return ggml_map_custom2(ctx, cond, uncond, &cfg_scale, out->ne[0], &model->cfg_scale_data);
}

static struct ggml_tensor * build_dia_encoder(ggml_context * ctx, dia_model * model, dia_context * dctx, dia_ubatch & batch) {
    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, model->max_encoder_context_length*2);
    ggml_set_input(dctx->inp_tokens);

    dctx->encode_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, model->max_encoder_context_length);
    ggml_set_input(dctx->encode_positions);

    struct ggml_tensor * attn_mask = build_dia_encoder_attn_mask(ctx, dctx, model);

    struct ggml_tensor * cur = ggml_reshape_3d(ctx, ggml_get_rows(ctx, model->encoder->embedding, dctx->inp_tokens), model->encoder_hidden_size, model->max_encoder_context_length, 2);
    for (auto layer : model->encoder->layers) {
        struct ggml_tensor * residual = cur;

        cur = dia_layer_norm(ctx, cur, layer->self_attn_norm);
        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer->q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer->k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer->v, cur);

            // Strangely Dia follows the neoX Rotary Positional Embeddings Protocol
            Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, model->head_size, model->encoder_attn_heads, model->max_encoder_context_length, 2)), dctx->encode_positions, model->head_size, 2);
            Kcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Kcur, model->head_size, model->encoder_attn_heads, model->max_encoder_context_length, 2)), dctx->encode_positions, model->head_size, 2);
            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, attn_mask, 1.0f, 0.0f);
            struct ggml_tensor * v = ggml_cont_4d(ctx, ggml_transpose(ctx, Vcur), model->max_encoder_context_length, model->head_size, model->encoder_attn_heads, 2);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);

            // It is unclear why the attention ops in Dia's encoder don't project to the embedding dimension size as is standard. Instead they up project to the decoder's embedding dimension
            // then down project back the the encoder embedding dimension.
            cur = ggml_cont_3d(ctx, kqv_merged, model->decoder_hidden_size, model->max_encoder_context_length, 2);
            cur = ggml_mul_mat(ctx, layer->o, cur);
        }

        cur = ggml_add(ctx, cur, residual);
        struct ggml_tensor * residual_mlp = cur;

        cur = dia_layer_norm(ctx, cur, layer->mlp_norm);
        // mlp
        {
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer->gate, cur)), ggml_mul_mat(ctx, layer->up, cur));
            cur = ggml_mul_mat(ctx, layer->out, cur);
        }

        cur = ggml_add(ctx, cur, residual_mlp);
    }

    cur = dia_layer_norm(ctx, cur, model->encoder->norm);
    return cur;
}

static struct ggml_tensor * repeat_interleave_dim1(ggml_context * ctx, struct ggml_tensor * a, int repeat) {
    //return ggml_repeat(ctx, a, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], 4*a->ne[1], a->ne[2], a->ne[3]));
    struct ggml_tensor * running;
    for (int i = 0; i < a->ne[1]; i++) {
        int offset = i * a->nb[1];
        struct ggml_tensor * t = ggml_cont(ctx, ggml_view_4d(ctx, a, a->ne[0], 1, a->ne[2], a->ne[3], a->nb[1], a->nb[2], a->nb[3], offset));
        t = ggml_repeat(ctx, t, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], repeat, a->ne[2], a->ne[3]));
        if (i == 0) {
            running = t;
        } else {
            running = ggml_concat(ctx, running, t, 1);
        }
    }
    return running;
}

static void build_dia_self_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv, ggml_cgraph * gf, struct ggml_tensor * k, struct ggml_tensor * v, dia_ubatch & batch, int layer_index) {
    int64_t attn_size = model->head_size * model->decoder_attn_heads;

    struct ggml_tensor * k_cache_view =
        ggml_view_2d(
                ctx, kv->k_l[layer_index], attn_size, 2,
                attn_size * model->max_generation_size * ggml_element_size(kv->k_l[layer_index]),
                attn_size*dctx->current_position*ggml_element_size(kv->k_l[layer_index]));

    k = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads / model->decoder_query_heads, batch.sequence_length, 2)), dctx->positions, model->head_size, 2);
    // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
    // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
    // from the cache
    k = repeat_interleave_dim1(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads / model->decoder_query_heads, batch.sequence_length, 2)), model->decoder_query_heads);
    k = ggml_cont(ctx, ggml_reshape_2d(ctx, k, attn_size, 2));

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_cache_view));

    struct ggml_tensor * v_cache_view = nullptr;

    v_cache_view = ggml_view_2d(
            ctx, kv->v_l[layer_index], attn_size, 2,
            attn_size * model->max_generation_size * ggml_element_size(kv->v_l[layer_index]),
            attn_size*dctx->current_position*ggml_element_size(kv->v_l[layer_index]));

    // Since the sequence length should always be 1 here this is the most pertinent time to repeat the heads for grouped query attention.
    // If GGML supported a repeat_interleave op then it would be more optimal to store just the groups in the cache and interleave the attention heads after recalling
    // from the cache
    v = repeat_interleave_dim1(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, v, model->head_size, model->decoder_attn_heads / model->decoder_query_heads, batch.sequence_length, 2)), model->decoder_query_heads);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_cache_view));
}

static void build_dia_cross_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv, ggml_cgraph * gf, struct ggml_tensor * encoder_hidden_states, int layer_index) {
    dia_decoder_layer * layer = model->decoder->layers[layer_index];
    struct ggml_tensor * encoder_states_key_view = ggml_cont(ctx, ggml_view_3d(
        ctx,
        encoder_hidden_states,
        model->encoder_hidden_size,
        dctx->prompt_size,
        2,
        model->encoder_hidden_size * ggml_element_size(encoder_hidden_states), model->encoder_hidden_size * model->max_encoder_context_length * ggml_element_size(encoder_hidden_states), 0));

    struct ggml_tensor * k = ggml_mul_mat(ctx, layer->cross_attn_k, encoder_states_key_view);
    struct ggml_tensor * positions_view = ggml_view_1d(ctx, dctx->encode_positions, dctx->prompt_size, 0);

    k = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, k, model->head_size, model->decoder_attn_heads, dctx->prompt_size, 2)), positions_view, model->head_size, 2);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 1, 3, 2));

    struct ggml_tensor * k_cache_view =
        ggml_view_4d(
                ctx, kv->cross_k_l[layer_index], model->head_size, model->decoder_attn_heads, 2, dctx->prompt_size,
                model->head_size*ggml_element_size(kv->cross_k_l[layer_index]),
                model->head_size*model->decoder_attn_heads*ggml_element_size(kv->cross_k_l[layer_index]),
                model->head_size*model->decoder_attn_heads*2*ggml_element_size(kv->cross_k_l[layer_index]),
                0);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_cache_view));

    struct ggml_tensor * v = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, layer->cross_attn_v, encoder_hidden_states)));
    v = ggml_cont_4d(ctx, v, model->max_encoder_context_length, model->head_size, model->decoder_attn_heads, 2);

    struct ggml_tensor * v_cache_view =
        ggml_view_4d(
                ctx, kv->cross_v_l[layer_index], model->max_encoder_context_length, model->head_size, model->decoder_attn_heads, 2,
                model->max_encoder_context_length*ggml_element_size(kv->cross_v_l[layer_index]),
                model->head_size*model->max_encoder_context_length*ggml_element_size(kv->cross_v_l[layer_index]),
                model->head_size*model->max_encoder_context_length*model->decoder_attn_heads*ggml_element_size(kv->cross_v_l[layer_index]),
                0);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_cache_view));
}

static struct ggml_tensor * build_dia_decoder(
        ggml_cgraph * gf,
        ggml_context * ctx,
        dia_model * model,
        dia_context * dctx,
        dia_kv_cache * cache,
        dia_ubatch & batch,
        struct ggml_tensor * encoder_hidden_states) {
    dctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(dctx->positions);
    struct ggml_tensor * cur = build_dia_decoder_inp_embd(ctx, dctx, model->decoder, batch, model->n_output_heads);

    for (int l = 0; l < model->decoder->layers.size(); l++){
        dia_decoder_layer * layer = model->decoder->layers[l];
        struct ggml_tensor * residual = cur;

        cur = dia_layer_norm(ctx, cur, layer->self_attn_norm);
        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer->self_attn_q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer->self_attn_k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer->self_attn_v, cur);

            build_dia_self_kv_store(ctx, dctx, model, cache, gf, Kcur, Vcur, batch, l);
            struct ggml_tensor * k =
                ggml_view_4d(ctx, cache->k_l[l],
                        model->head_size, model->decoder_attn_heads, dctx->current_position + 1, 2,
                        ggml_element_size(cache->k_l[l]) * model->head_size,
                        ggml_element_size(cache->k_l[l]) * model->decoder_attn_heads * model->head_size,
                        ggml_element_size(cache->k_l[l]) * model->decoder_attn_heads * model->head_size * model->max_generation_size,
                        0);
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

            struct ggml_tensor * v =
                ggml_view_3d(ctx, cache->v_l[l],
                        model->head_size * model->decoder_attn_heads, dctx->current_position + 1, 2,
                        ggml_element_size(cache->v_l[l]) * model->decoder_attn_heads * model->head_size,
                        ggml_element_size(cache->v_l[l]) * model->decoder_attn_heads * model->head_size * model->max_generation_size,
                        0);
            v = ggml_cont_4d(ctx, ggml_transpose(ctx, v), dctx->current_position + 1, model->head_size, model->decoder_attn_heads, 2);

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Qcur, model->head_size, model->decoder_attn_heads, batch.sequence_length, 2)), dctx->positions, model->head_size, 2);
            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            struct ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);

            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, kqv_merged, model->decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer->self_attn_o, cur);
        }


        // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
        cur = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        cur = ggml_add(ctx, cur, residual);
        struct ggml_tensor * residual_cross = cur;

        cur = dia_layer_norm(ctx, cur, layer->cross_attn_norm);
        // cross-attention
        {
            struct ggml_tensor * cross_Qcur = ggml_mul_mat(ctx, layer->cross_attn_q, cur);

            // only load the cross attention kv store when performing the encoding step
            if (batch.encoder_step) {
                build_dia_cross_kv_store(ctx, dctx, model, cache, gf, encoder_hidden_states, l);
            }

            struct ggml_tensor * cross_k =
                ggml_view_4d(
                        ctx, cache->cross_k_l[l], model->head_size, model->decoder_attn_heads, 2,
                        model->max_encoder_context_length, model->head_size*ggml_element_size(cache->cross_k_l[l]),
                        model->head_size*model->decoder_attn_heads*ggml_element_size(cache->cross_k_l[l]),
                        model->head_size*model->decoder_attn_heads*2*ggml_element_size(cache->cross_k_l[l]),
                        0);
            // the double permute operation shouldn't be necessary here, but it seems that currently ggml permute only currently alows for a single
            // axis pair to be transposed.
            cross_k = ggml_cont(ctx, ggml_permute(ctx, ggml_permute(ctx, cross_k, 0, 1, 3, 2), 0, 2, 1, 3));

            struct ggml_tensor * cross_v =
                ggml_cont(ctx, ggml_view_4d(
                        ctx, cache->cross_v_l[l], model->max_encoder_context_length, model->head_size, model->decoder_attn_heads, 2,
                        model->max_encoder_context_length*ggml_element_size(cache->cross_v_l[l]),
                        model->head_size*model->max_encoder_context_length*ggml_element_size(cache->cross_v_l[l]),
                        model->head_size*model->max_encoder_context_length*model->decoder_attn_heads*ggml_element_size(cache->cross_v_l[l]),
                        0));

            // As noted in the encoder Dia uses the Neo-X protocol for RoPE.
            cross_Qcur = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, cross_Qcur, model->head_size, model->decoder_attn_heads, batch.sequence_length, 2)), dctx->positions, model->head_size, 2);
            struct ggml_tensor * cross_q = ggml_cont(ctx, ggml_permute(ctx, cross_Qcur, 0, 2, 1, 3));
            struct ggml_tensor * cross_kq = ggml_mul_mat(ctx, cross_k, cross_q);

            // given that attention bias, scaling and masking are not used for decoding, it might be faster to prefer the #ggml_soft_max op here,
            cross_kq = ggml_soft_max_ext(ctx, cross_kq, nullptr, 1.0f, 0.0f);
            struct ggml_tensor * cross_kqv = ggml_mul_mat(ctx, cross_kq, cross_v);
            struct ggml_tensor * cross_kqv_merged = ggml_cont(ctx, ggml_permute(ctx, cross_kqv, 2, 0, 1, 3));
            cur = ggml_cont_3d(ctx, cross_kqv_merged, model->decoder_hidden_size, batch.sequence_length, 2);
            cur = ggml_mul_mat(ctx, layer->cross_attn_o, cur);
        }


        // if we ever need to support multiple step decoder runs then this reshape will need to be replaced with permutation.
        cur = ggml_cont_2d(ctx, cur, cur->ne[0], 2);
        cur = ggml_add(ctx, cur, residual_cross);
        struct ggml_tensor * residual_mlp = cur;

        cur = dia_layer_norm(ctx, cur, layer->mlp_norm);
        // mlp
        {
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer->gate, cur)), ggml_mul_mat(ctx, layer->up, cur));
            cur = ggml_mul_mat(ctx, layer->out, cur);
        }

        cur = ggml_add(ctx, cur, residual_mlp);
    }

    cur = dia_layer_norm(ctx, cur, model->decoder->norm);
    cur = build_dia_head_outputs(ctx, model, cur);
    return cur;
}

void dia_runner::tokenize_sentence(std::string sentence, dia_ubatch & batch) {
    // Dia's tokenization process is unusual. Essentially Dia takes the byte value for each character and uses that as
    // a token array. Additionally, because Dia performs a cfg-scale adjustment before sampling tokens, it is necessary to
    // generate with a conditioned context (i.e. with the text) and an unconditioned context (i.e. without any text) so that
    // proper adjustments can be perfored at each generation step. This means that we need to pad the end of our tokens to the
    // max context size for both the conditional and unconditional sequence.

    // if the sentence isn't prepended by dialogue start tokens, [S1] or [S2], then append one.
    sentence = strip(sentence);
    std::string start = sentence.substr(0, 4);
    if (start != "[S1]" && start != "[S2]") {
        sentence = "[S1] " + sentence;
    }
    if (sentence[sentence.size() - 1] != '.') {
        sentence += ".";
    }

    // [S1] and [S2] are special character sequences that are replaced with the special tokens 0x01 and 0x02 respectively.
    std::string r1(1, 1);
    std::string r2(1, 2);
    while (sentence.find("[S1]") != std::string::npos) {
        size_t pos = sentence.find("[S1]");
        sentence.replace(pos, 4, r1);
    }
    while (sentence.find("[S2]") != std::string::npos) {
        size_t pos = sentence.find("[S2]");
        sentence.replace(pos, 4, r2);
    }

    if (sentence.size() > model->max_encoder_context_length) {
        TTS_ABORT("Dia currently only supports a max of %d characters and received an input of %d characters.", model->max_encoder_context_length, sentence.size());
    }
    batch.tokens.reserve(model->max_encoder_context_length * 2);
    for (auto character : sentence) {
        batch.tokens.push_back((uint32_t) character);
    }
    batch.sentence_length = batch.tokens.size();
    // this 100 token warning is arbitrarily chosen based on spot checking small prompt performance
    if (batch.sentence_length <= 100) {
        fprintf(stdout, "Your prompt has fewer than 100 tokens. Please note that Dia's generation with prompts that are fewer than 100 tokens is highly inconsistent.\n");
    }

    for (int i = (int) batch.tokens.size(); i < model->max_encoder_context_length * 2; i++) {
        batch.tokens.push_back(0u);
    }
 }

dia_ubatch dia_runner::batch_from_sentence(std::string sentence) {
    // if we are generating a new batch from tokens then we need to run the encoder step;
    struct dia_ubatch batch{ 1, true};
    tokenize_sentence(sentence, batch);
    batch.audio_tokens.reserve(model->n_output_heads);
    for (int i = 0; i < model->n_output_heads; i++) {
        batch.audio_tokens.push_back(model->bos_token_id);
    }
    return batch;
}

/*
 * There are two unique features of Dia's model architecture:
 * 1.  Dia cleans its output generation by adding the difference between its text based output (its conditional output) and its unconditional output
 *     to the conditional ouput before sampling. This is why the batch is set to two throughout the graph.
 *
 * 2.  Dia's decoder attends across the entire encoded space including the pad buffer which receives a unique attention mask. This is why the
 *     encoder sequence is always max length.
 */
struct ggml_cgraph * dia_runner::build_dia_graph(dia_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    struct ggml_tensor * encoded_states = nullptr;

    if (batch.encoder_step) {
        encoded_states = build_dia_encoder(ctx, model, dctx, batch);
        ggml_build_forward_expand(gf, encoded_states);
    }

    struct ggml_tensor * cur = build_dia_decoder(gf, ctx, model, dctx, kv_cross_self, batch, encoded_states);
    ggml_set_name(cur, "decoder_output");
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}

void dia_runner::configure_generation(generation_configuration * config) {
    GGML_ASSERT(config->max_tokens == 0 || config->max_tokens > model->max_delay);
    decode_sampler->temperature = config->temperature;
    decode_sampler->repetition_penalty = config->repetition_penalty;
    decode_sampler->do_sample = config->sample;
    decode_sampler->top_k = config->top_k;
    decode_sampler->top_p = config->top_p;
    dctx->max_generation_size = config->max_tokens > model->max_delay ? config->max_tokens : model->max_generation_size;
}

void dia_runner::set_inputs(dia_ubatch & batch) {
    if (batch.encoder_step) {
        ggml_backend_tensor_set(dctx->inp_tokens, batch.tokens.data(), 0, batch.tokens.size()*ggml_element_size(dctx->inp_tokens));
        int32_t * ep = (int32_t*) dctx->encode_positions->data;
        float * mask = (float*) dctx->encode_attn_mask->data;
        for (int i = 0; i < model->max_encoder_context_length; i++) {
            ep[i] = (int32_t) i;
            for (int ii = 0; ii < model->max_encoder_context_length; ii++) {
                if (i < batch.sentence_length) {
                    mask[i*model->max_encoder_context_length + ii] = ii < batch.sentence_length ? 0.0 : -INFINITY;
                } else {
                    mask[i*model->max_encoder_context_length + ii] = ii >= batch.sentence_length ? 0.0 : -INFINITY;
                }
            }
        }
    }
    // The audio tokens need to be repeated in the input in order to support cfg-scaling. I.E we need duplicate inputs for conditional and unconditional logits.
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(), 0, batch.audio_tokens.size()*ggml_element_size(dctx->audio_inp_tokens));
    ggml_backend_tensor_set(dctx->audio_inp_tokens, batch.audio_tokens.data(), batch.audio_tokens.size()*ggml_element_size(dctx->audio_inp_tokens), batch.audio_tokens.size()*ggml_element_size(dctx->audio_inp_tokens));
    ((int32_t*) dctx->positions->data)[0] = dctx->current_position;
}

int dia_runner::decode(dia_ubatch & batch) {
    if (batch.encoder_step) {
        dctx->prompt_size = batch.sentence_length;
        dctx->output_tokens.reserve(dctx->max_generation_size * model->n_output_heads);
    }
    ggml_backend_sched_reset(dctx->sched);

    const size_t logits_size = model->output_vocab_size * dctx->max_generation_size * model->n_output_heads;
    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size  = logits_size * sizeof(float);

    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }

    dctx->logits = (float *) ggml_backend_buffer_get_base(dctx->buf_output);

    ggml_cgraph * gf = build_dia_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    std::string resname = ggml_get_name(res);
    ggml_backend_sched_alloc_graph(dctx->sched, gf);

    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    float * logits_out = dctx->logits + dctx->current_position * model->output_vocab_size * model->n_output_heads;
    dctx->get_ggml_node_data(res, logits_out, model->output_vocab_size * model->n_output_heads * sizeof(float));

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);

    return 0;
}

dia_ubatch dia_runner::build_worst_case_batch()  {
    struct dia_ubatch batch{ 1, true };
    batch.tokens.resize(model->max_encoder_context_length * 2);
    batch.audio_tokens.resize(model->n_output_heads);
    return batch;
}

void dia_runner::prepare_post_load() {
    dac_runner->prepare_post_load();
    dia_kv_cache_init(kv_cross_self, model, dctx);
    auto batch = build_worst_case_batch();
    batch.sentence_length = model->max_encoder_context_length;
    dctx->prompt_size = model->max_encoder_context_length;
    auto gf = build_dia_graph(batch);
    dctx->prep_schedule(gf);
}

bool dia_runner::check_stopping(dia_ubatch & batch) {
    if (dctx->delay_steps == -1 && (batch.audio_tokens[0] == model->eos_token_id || dctx->current_position >= dctx->max_generation_size - model->max_delay)) {
        dctx->delay_steps = model->max_delay;
    }

    if (dctx->delay_steps > 0) {
        int step_after_eos = model->max_delay - dctx->delay_steps;
        for (int i = 0; i < model->delay_pattern.size(); i++) {
            if (step_after_eos == model->delay_pattern[i]) {
                batch.audio_tokens[i] = model->eos_token_id;
            } else if (step_after_eos > model->delay_pattern[i]) {
                batch.audio_tokens[i] = model->pad_token_id;
            }
        }
        dctx->delay_steps -= 1;
    }
    return dctx->delay_steps == 0;
}

void dia_runner::adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < (size / model->n_output_heads) - model->max_delay; i++) {
        bool skip_step = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i*model->n_output_heads+model->delay_pattern[ii]*model->n_output_heads+ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                skip_step = true;
                break;
            }
        }
        if (!skip_step) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i*model->n_output_heads+model->delay_pattern[ii]*model->n_output_heads+ii;
                filtered.push_back(output_tokens[next_index]);
            }
        }
    }
}

int dia_runner::generate_from_batch(dia_ubatch & batch, struct tts_response * output) {
    while (!check_stopping(batch)) {
        int state = decode(batch);
        if (state != 0) {
            return state;
        }
        decode_sampler->sample(dctx->logits + dctx->current_position * model->n_output_heads * model->output_vocab_size, dctx->output_tokens);
        dctx->current_position += batch.sequence_length;
        batch = dia_ubatch{ 1 };
        uint32_t * last_outputs = (dctx->output_tokens.data() + (int) dctx->output_tokens.size() - model->n_output_heads);
        batch.audio_tokens.reserve(model->n_output_heads);
        for (int i = 0; i < model->n_output_heads; i++) {
            batch.audio_tokens.push_back(dctx->current_position > i ? last_outputs[i] : model->bos_token_id);
        }
    }

    std::vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(dctx->output_tokens, filtered_output_tokens);

    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads, output);
    return 0;
}

int dia_runner::generate(std::string sentence, struct tts_response * output) {
    dia_ubatch batch = batch_from_sentence(sentence);
    dctx->reset();
    decode_sampler->reset();
    dctx->current_position = 0;
    if (!kv_cross_self) {
        kv_cross_self = new dia_kv_cache;
        if (!dia_kv_cache_init(kv_cross_self, model, dctx)) {
            return 1;
        }
    }
    return generate_from_batch(batch, output);
}

void dia_runner::assign_weight(std::string name, ggml_tensor * tensor) {
    if (tensor->data == NULL) {
        return;
    }

    if (name.size() == 0) {
        // handles the top level meta tensor
        return;
    }

    if (name.size() > 14 && name.substr(0, 14) == "audio_encoder.") {
        dac_runner->model->assign_weight(name.substr(14), tensor);
    } else {
        model->assign_weight(name, tensor);
    }
}

#include "orpheus_model.h"

#include <array>

// These tokens and variables aren't defined in the Orpheus' model configuration but instead are defined inline in various python functions.
// As such, they are not discoverable so defining them as unconfigurable constants should be fine.
static constexpr std::array<const char *, 7> orpheus_voices{"zoe", "zac","jess", "leo", "mia", "julia", "leah"};
static constexpr std::array<uint32_t, 2> orpheus_prepended_tokens = { 128259, 128000 };
static constexpr std::array<uint32_t, 4> orpheus_appended_tokens = { 128009, 128260, 128261, 128257 };

void orpheus_model::assign_weight(std::string name, struct ggml_tensor * tensor) {
    if (name == "norm") {
        output_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(output_norm, tensor);
    } else if (name == "lm_head") {
        head = ggml_dup_tensor(ctx, tensor);
        set_tensor(head, tensor);
    } else if (name == "embed_tokens") {
        embd = ggml_dup_tensor(ctx, tensor);
        set_tensor(embd, tensor);
    } else if (name == "rope_frequencies") {
        rope_frequencies = ggml_dup_tensor(ctx, tensor);
        set_tensor(rope_frequencies, tensor);
    } else if (has_prefix(name, "layers")) {
        auto lpair = parse_layer_count(name);
        int l = lpair.first;
        std::string lt_name = lpair.second;
        assign_to_layer(lt_name, layers[l], tensor);
    }
}

void orpheus_model::assign_to_layer(std::string part, orpheus_layer & layer, struct ggml_tensor * tensor) {
    if (part == ".self_attn.k_proj") {
        layer.k = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.k, tensor);
    } else if (part == ".self_attn.q_proj") {
        layer.q = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.q, tensor);
    } else if (part == ".self_attn.v_proj") {
        layer.v = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.v, tensor);
    } else if (part == ".self_attn.o_proj") {
        layer.o = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.o, tensor);
    } else if (part == ".mlp.gate_proj") {
        layer.gate = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.gate, tensor);
    } else if (part == ".mlp.up_proj") {
        layer.up = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.up, tensor);
    } else if (part == ".mlp.down_proj") {
        layer.down = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.down, tensor);
    } else if (part == ".input_layernorm") {
        layer.input_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.input_norm, tensor);
    } else if (part == ".post_attention_layernorm") {
        layer.post_attention_norm = ggml_dup_tensor(ctx, tensor);
        set_tensor(layer.post_attention_norm, tensor);
    }
}

void orpheus_model::prep_constants(gguf_context * meta) {
    // get constants for orpheus
    int vocab_size_key = gguf_find_key(meta, "orpheus.vocab_size");
    if (vocab_size_key != -1) {
        vocab_size = gguf_get_val_u32(meta, vocab_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "orpheus.attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }

    int kv_attn_heads_key = gguf_find_key(meta, "orpheus.kv_attn_heads");
    if (kv_attn_heads_key != -1) {
        n_kv_attn_heads = gguf_get_val_u32(meta, kv_attn_heads_key);
    }

    int head_size_key = gguf_find_key(meta, "orpheus.head_dim");
    if (head_size_key != -1) {
        head_size = gguf_get_val_u32(meta, head_size_key);
    }

    int stopping_token_key = gguf_find_key(meta, "orpheus.stopping_token_id");
    if (stopping_token_key != -1) {
        stopping_token_id = gguf_get_val_u32(meta, stopping_token_key);;
    }

    int eos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int hidden_size_key = gguf_find_key(meta, "orpheus.hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int kv_hidden_size_key = gguf_find_key(meta, "orpheus.kv_hidden_size");
    if (kv_hidden_size_key != -1) {
        kv_hidden_size = gguf_get_val_u32(meta, kv_hidden_size_key);
    }
}

void orpheus_model::prep_layers(gguf_context * meta) {
    int n_layers_key = gguf_find_key(meta, "orpheus.layers");
    if (n_layers_key == -1) {
        TTS_ABORT("the 'orpheus.layers' must be specified in the GGUF file.");
    }
    n_layers = (int) gguf_get_val_u32(meta, n_layers_key);
    for (int i = 0; i < n_layers; i++) {
        layers.push_back(orpheus_layer{});
    }
}

struct ggml_tensor * orpheus_build_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight) {
    float eps = 0.00001;
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

struct ggml_tensor * build_attn_mask(ggml_context * ctx, orpheus_context * octx, orpheus_ubatch & batch) {
    octx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) octx->current_position + batch.n_tokens, (int64_t) octx->current_position + batch.n_tokens);
    ggml_set_input(octx->attn_mask);
    return octx->attn_mask;
}

 void orpheus_context::reset() {
    output_tokens.clear();
    current_position = 0;
    n_outputs = 0;
 }

orpheus_context * build_new_orpheus_context(orpheus_model * model, int n_threads, bool use_cpu) {
    orpheus_context * octx = new orpheus_context(model, n_threads);
    octx->backend_cpu = ggml_backend_cpu_init();
    octx->set_threads();
    octx->build_schedule();
    octx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return octx;
}

void orpheus_runner::orpheus_kv_cache_init() {
    ggml_backend_buffer_type_t buft = nullptr;
    if (octx->backend != nullptr) {
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ (2u * model->layers.size() + 1)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        TTS_ABORT("%s: failed to initialze ggml context for key value cache.\n", __func__);
    }
    if (!kv_self) {
        kv_self = new orpheus_kv_cache;
    }
    kv_self->ctx = ctx;
    kv_self->k_l.reserve(model->layers.size());
    kv_self->v_l.reserve(model->layers.size());

    for (int i = 0; i < (int) model->layers.size(); i++) {
        ggml_tensor * k = ggml_new_tensor_1d(kv_self->ctx, kv_self->cache_type, model->hidden_size * (model->max_context_length + model->max_generation_size));
        ggml_tensor * v = ggml_new_tensor_1d(kv_self->ctx, kv_self->cache_type, model->hidden_size * (model->max_context_length + model->max_generation_size));
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        kv_self->k_l.push_back(k);
        kv_self->v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(kv_self->ctx, buft);
    ggml_backend_buffer_clear(buf, 0);
    kv_self->buf = buf;
 }

 void orpheus_runner::orpheus_build_kv_store(struct ggml_context * ctx, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int index, uint32_t n_tokens, int repeat) {
    k_cur = ggml_rope_ext(ctx, ggml_cont(ctx, ggml_reshape_3d(ctx, k_cur, model->head_size, model->n_kv_attn_heads, n_tokens)), octx->positions, model->rope_frequencies,
                model->head_size, 2,0, 500000.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // A performance comparison between this method, i.e. performing 3 incremental copy operations in order to achieve repeat_interleave,
    // and performing the repeat operation upfront before performign a single copy needs to be performed in order to better optimize this function.
    // Additionally, it might be more performant for the values transposition to be performed prior to appending it to the cache, as it would save us
    // from incrementally larger transpositions with generation.
    for (int i = 0; i < repeat; i++) {
        struct ggml_tensor * k_cache_view = ggml_view_3d(
            ctx,
            kv_self->k_l[index],
            model->head_size,
            model->n_kv_attn_heads,
            n_tokens,
            ggml_element_size(kv_self->k_l[index]) * model->head_size * repeat,
            ggml_element_size(kv_self->k_l[index]) * model->n_attn_heads * model->head_size,
            ggml_element_size(kv_self->k_l[index]) * model->n_attn_heads * model->head_size * octx->current_position + i * ggml_element_size(kv_self->k_l[index]) * model->head_size
        );
        ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

        struct ggml_tensor * v_cache_view = ggml_view_3d(
            ctx,
            kv_self->v_l[index],
            model->head_size,
            model->n_kv_attn_heads,
            n_tokens,
            ggml_element_size(kv_self->k_l[index]) * model->head_size * repeat,
            ggml_element_size(kv_self->k_l[index]) * model->n_attn_heads * model->head_size,
            ggml_element_size(kv_self->k_l[index]) * model->n_attn_heads * model->head_size * octx->current_position + i * ggml_element_size(kv_self->k_l[index]) * model->head_size
        );
        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
    }
}

struct ggml_cgraph * orpheus_runner::build_orpheus_graph(orpheus_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    const int32_t full_sequence_length = octx->current_position + (uint32_t) batch.n_tokens;
    octx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(octx->positions);
    octx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(octx->inp_tokens);
    inpL = ggml_get_rows(ctx, model->embd, octx->inp_tokens);

    struct ggml_tensor * KQ_mask_dec = build_attn_mask(ctx, octx, batch);

    for (int l = 0; l < model->n_layers; l++) {
        struct ggml_tensor * residual = inpL;
        cur = orpheus_build_layer_norm(ctx, inpL, model->layers[l].input_norm);

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l].q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l].k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l].v, cur);

            orpheus_build_kv_store(ctx, gf, Kcur, Vcur, l, batch.n_tokens, 3);
            struct ggml_tensor * k =
                ggml_cont(ctx, ggml_view_3d(ctx, kv_self->k_l[l],
                        model->head_size, full_sequence_length, model->n_attn_heads,
                        ggml_element_size(kv_self->k_l[l]) * model->n_attn_heads * model->head_size,
                        ggml_element_size(kv_self->k_l[l]) * model->head_size,
                        0));

            struct ggml_tensor * v =
                ggml_view_2d(ctx, kv_self->v_l[l],
                        model->hidden_size, full_sequence_length,
                        ggml_element_size(kv_self->k_l[l]) * model->hidden_size,
                        0);

            v = ggml_cont_3d(ctx, ggml_transpose(ctx, v), full_sequence_length, model->head_size, model->n_attn_heads);

            Qcur = ggml_rope_ext(
                ctx, ggml_cont(ctx, ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens)),
                octx->positions, model->rope_frequencies, model->head_size, 2, 0, 500000.0f, // rope theta
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f/sqrtf(model->head_size), 0.0f);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
            attn_out = ggml_mul_mat(ctx, model->layers[l].o, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);

        struct ggml_tensor * residualffn = cur;

        // mlp
        {
            cur = orpheus_build_layer_norm(ctx, cur, model->layers[l].post_attention_norm);
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, model->layers[l].gate, cur)), ggml_mul_mat(ctx, model->layers[l].up, cur));
            cur = ggml_mul_mat(ctx, model->layers[l].down, cur);
        }
        cur = ggml_add(ctx, cur, residualffn);
        inpL = cur;
    }

    cur = orpheus_build_layer_norm(ctx, cur, model->output_norm);
    // only about 40k of the output head is actually uses for generation purposes. Ideally the head tensor should be shrunk and sampled tokens should be incremented.
    cur = ggml_mul_mat(ctx, model->head, cur);
    if (batch.n_tokens > 1) {
        cur = ggml_cont(ctx, ggml_view_1d(ctx, cur, model->vocab_size, ggml_element_size(cur) * (cur->ne[1] - 1) * model->vocab_size));
    }
    ggml_build_forward_expand(gf, cur);
    free_build();

    return gf;
}

void orpheus_runner::decode(orpheus_ubatch & batch) {
    ggml_backend_sched_reset(octx->sched);

    octx->output_tokens.reserve(model->max_generation_size);

    const size_t new_size  = model->vocab_size * model->max_generation_size * sizeof(float);
    octx->prep_output_buffer(new_size);

    ggml_cgraph * gf = build_orpheus_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(octx->sched, gf);

    set_inputs(batch);
    ggml_backend_sched_graph_compute_async(octx->sched, gf);

    float * logits_out = octx->logits + octx->n_outputs * model->vocab_size;
    octx->get_ggml_node_data(res, logits_out, model->vocab_size * sizeof(float));

    // update the total number of outputs retrieved and the current position
    octx->current_position += batch.n_tokens;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(octx->sched);
}

void orpheus_runner::set_inputs(orpheus_ubatch & batch) {
    ggml_backend_tensor_set(octx->inp_tokens, batch.tokens.data(), 0, batch.tokens.size()*ggml_element_size(octx->inp_tokens));
    int32_t * pos = (int32_t*) octx->positions->data;
    float * mask = (float*) octx->attn_mask->data;
    uint32_t max_pos = octx->current_position + batch.n_tokens;
    for (int i = 0; i < batch.n_tokens; i++) {
        pos[i] = (int32_t) octx->current_position + i;
        for (int ii = 0; ii < max_pos; ii++) {
            mask[i*max_pos + ii] = ii > pos[i] ? -INFINITY : 0.0f;
        }
    }
}

orpheus_ubatch orpheus_runner::batch_from_sentence(std::string sentence) {
    struct orpheus_ubatch batch;
    for (auto t : orpheus_prepended_tokens) {
        batch.tokens.push_back(t);
    }
    if (!octx->voice.empty()) {
        sentence = octx->voice  + ": " + sentence;
    }
    tokenizer->tokenize(sentence, batch.tokens);
    for (auto t : orpheus_appended_tokens) {
        batch.tokens.push_back(t);
    }
    batch.n_tokens = batch.tokens.size();
    return batch;
}

std::vector<std::vector<uint32_t>> orpheus_runner::prepare_output_tokens() {
    size_t chunks = octx->output_tokens.size() / 7;
    std::vector<std::vector<uint32_t>> output_tokens;
    for (int i = 0; i < model->audio_heads; i++) {
        output_tokens.push_back(std::vector<uint32_t>{});
    }
    for (int i = 0; i < chunks; i++) {
        for (int ii = 0; ii < 7; ii++) {
            uint32_t thead = model->heads[ii];
            // the manipulations below are not configured because they are performed inline via undocumented constants in the Orpheus codebase.
            // Essentially this is how Orpheus converts discrete samples from the output shape to the audio input shape.
            uint32_t t = octx->output_tokens[i*7 + ii] - 128266 - ((ii % 7) * 4096);
            output_tokens[thead].push_back(t);
        }
    }
    return output_tokens;
}

void orpheus_runner::generate_from_batch(orpheus_ubatch & batch, struct tts_response * output) {
    while ((octx->output_tokens.size() == 0 || octx->output_tokens.back() != model->stopping_token_id) && octx->output_tokens.size() < model->max_generation_size) {
        decode(batch);
        generation_sampler->sample(octx->logits + octx->n_outputs * model->vocab_size, octx->output_tokens);
        // only increment the output count after sampling
        octx->n_outputs++;
        batch = orpheus_ubatch{
            1, {octx->output_tokens.back()}
        };
    }
    // this case could be better addressed by adding spliting to the generation process.
    if (octx->output_tokens.size() >= model->max_generation_size) {
        fprintf(stdout, "Warning: generation hit its max default length. The generated audio may not contain the entire prompt.\n");
    }
    std::vector<std::vector<uint32_t>> processed_output_tokens = prepare_output_tokens();
    srunner->run(processed_output_tokens, output);
}

int orpheus_runner::generate(std::string sentence, struct tts_response * response) {
    orpheus_ubatch batch = batch_from_sentence(sentence);
    // it should be possible to update the max context window size, but currently it is extremely unlikely that a single prompt will
    // surpass the default size.
    if (batch.tokens.size() > model->max_context_length) {
        fprintf(stdout,"The prompt was too large for the default context window. Try splitting up or shortenning the prompt.");
        return -1;
    }
    octx->reset();
    generation_sampler->reset();
    if  (!kv_self) {
        orpheus_kv_cache_init();
    }
    generate_from_batch(batch, response);
    return 0;
}

void orpheus_runner::configure_generation(generation_configuration * config) {
    generation_sampler->temperature = config->temperature;
    generation_sampler->repetition_penalty = config->repetition_penalty;
    generation_sampler->do_sample = config->sample;
    generation_sampler->top_k = config->top_k;
    generation_sampler->top_p = config->top_p;
    if (std::find(orpheus_voices.begin(), orpheus_voices.end(), config->voice) == orpheus_voices.end() && !config->voice.empty()) {
        fprintf(stdout,"Voice '%s' is not a valid voice for Orpheus. Defaulting to zoe.", config->voice.c_str());
        config->voice = "zoe";
    }
    octx->voice = config->voice;
}

orpheus_ubatch orpheus_runner::build_worst_case_batch() {
    orpheus_ubatch batch;
    batch.n_tokens = model->max_context_length;
    return batch;
}

void orpheus_runner::assign_weight(std::string name, ggml_tensor * tensor) {
    if (tensor->data == NULL) {
        return;
    }

    if (name.size() == 0) {
        // handles the top level meta tensor
        return;
    }

    if (name.size() > 5 && name.substr(0, 5) == "snac.") {
        srunner->model->assign_weight(name.substr(5), tensor);
    } else if (name.size() > 8 && name.substr(0, 8) == "orpheus.") {
        model->assign_weight(name.substr(8), tensor);
    } else {
        fprintf(stdout, "Warning: function %s encountered an unhandled tensor named '%s'.\n", __func__, name.c_str());
    }
}

void orpheus_runner::prepare_post_load() {
    srunner->prepare_post_load();
    orpheus_kv_cache_init();
    auto batch = build_worst_case_batch();
    auto gf = build_orpheus_graph(batch);
    octx->prep_schedule(gf);
}

std::vector<std::string> list_voices() {
	std::vector<std::string> voices;
	voices.reserve(orpheus_voices.size());
	for (auto voice : orpheus_voices) {
		voices.push_back(voice);
	}
	return voices;
}

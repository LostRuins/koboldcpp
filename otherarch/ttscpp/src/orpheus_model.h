#pragma once

#include "ttssampler.h"
#include "ttstokenizer.h"
#include "snac_model.h"

// Orpheus uses vLLM with a llama-3 architecture. The only critical difference from the normal llama architecture is the use of kv heads.

struct orpheus_layer {
    struct ggml_tensor * input_norm;
    struct ggml_tensor * post_attention_norm;
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    struct ggml_tensor * o;
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * down;
};

struct orpheus_model : tts_model {
    uint32_t vocab_size = 156940;
    uint32_t n_attn_heads = 24;
    uint32_t n_kv_attn_heads = 8;
    uint32_t head_size = 128;
    uint32_t max_context_length = 1024;
    // the generation size is technically arbitrary as the model can handle a large context. This size comes out to being 25.6 seconds.
    uint32_t max_generation_size = 2100;
    uint32_t stopping_token_id = 128258;
    uint32_t eos_token_id = 128001;
    uint32_t bos_token_id = 128000;
    uint32_t hidden_size = 3072;
    uint32_t kv_hidden_size = 1024;
    uint32_t audio_heads = 3;
    uint32_t heads[7] = {0, 1, 2, 2, 1, 2, 2};

    int n_layers = 28;

    struct std::vector<orpheus_layer> layers;
    struct ggml_tensor * head;
    struct ggml_tensor * embd;
    struct ggml_tensor * output_norm;
    struct ggml_tensor * rope_frequencies;

    void assign_weight(std::string name, ggml_tensor * tensor);
    void assign_to_layer(std::string part, orpheus_layer & layer, struct ggml_tensor * tensor);
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "orpheus", 1.30);
    }
};

struct orpheus_context : runner_context {
    orpheus_context(orpheus_model * model, int n_threads): runner_context(n_threads), model(model) {};
    struct orpheus_model * model;

    uint32_t current_position = 0; // current position in the active sequence
    uint32_t n_outputs = 0; // the position of the text prompt termination (used for adjusting the cache when incrementally generating)
    std::string voice;

    std::vector<uint32_t> output_tokens;

    void reset();
    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * attn_mask;
    struct ggml_tensor * positions;
};

struct orpheus_kv_cache {
    ggml_type cache_type = GGML_TYPE_F32;

    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;

    struct ggml_context * ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t buf;

    void free() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }

    ~orpheus_kv_cache() {
        free();
    }
};

struct orpheus_context * build_new_orpheus_context(struct orpheus_model * model, int n_threads, bool use_cpu = true);

struct orpheus_ubatch {
    orpheus_ubatch() = default;
    orpheus_ubatch(size_t n_tokens, std::vector<uint32_t> tokens): n_tokens(n_tokens), tokens(tokens) {};
    size_t n_tokens; // total sentence tokens
    std::vector<uint32_t> tokens;    // [n_tokens]
};

struct orpheus_runner : tts_runner {
    orpheus_runner(
            orpheus_model * model,
            snac_runner * audio_decoder,
            orpheus_context * octx,
            bpe_tokenizer * bt,
            sampler * samp,
            orpheus_kv_cache * cache): model(model), srunner(audio_decoder), octx(octx), tokenizer(bt), generation_sampler(samp), kv_self(cache) {
        tts_runner::sampling_rate = 24000.0f;
        generation_sampler->n_output_heads = 1;
        generation_sampler->vocab_size = model->vocab_size;
        generation_sampler->eos_token_id = model->eos_token_id;
    }
    orpheus_model * model;
    snac_runner * srunner;
    orpheus_context * octx;
    bpe_tokenizer * tokenizer;
    orpheus_kv_cache * kv_self;
    sampler * generation_sampler;

    void init_build() {
        tts_runner::init_build(&octx->buf_compute_meta);
    }

    std::vector<std::string> list_voices();
    struct ggml_cgraph * build_orpheus_graph(orpheus_ubatch & batch);
    void orpheus_kv_cache_init();
    void orpheus_build_kv_store(struct ggml_context * ctx, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int index, uint32_t n_tokens, int repeat);
    void configure_generation(generation_configuration * config);
    void assign_weight(std::string name, ggml_tensor * tensor);
    std::vector<std::vector<uint32_t>> prepare_output_tokens();
    orpheus_ubatch build_worst_case_batch();
    orpheus_ubatch batch_from_sentence(std::string sentence);
    void set_inputs(orpheus_ubatch & batch);
    void decode(orpheus_ubatch & batch);
    void prepare_post_load();
    int generate(std::string sentence, struct tts_response * response);
    void generate_from_batch(orpheus_ubatch & batch, struct tts_response * output);
};

static struct ggml_tensor * orpheus_build_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight);
static struct ggml_tensor * build_attn_mask(ggml_context * ctx, orpheus_context * octx, orpheus_ubatch & batch);

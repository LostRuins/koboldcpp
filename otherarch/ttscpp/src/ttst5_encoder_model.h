#ifndef t5_encoder_model_h
#define t5_encoder_model_h

#include "tts_model.h"
#include "ttstokenizer.h"


enum t5_tensor {
    T5_EMBD,
    T5_NORM,
    T5_DOWN_PROJ,
    T5_DOWN_PROJ_BIAS,
    T5_RELATIVE_BIAS,
    T5_LAYER_ATTN_Q,
    T5_LAYER_ATTN_K,
    T5_LAYER_ATTN_V,
    T5_LAYER_ATTN_O,
    T5_LAYER_ATTN_NORM,
    T5_LAYER_WI_0,
    T5_LAYER_WI_1,
    T5_LAYER_WO,
    T5_LAYER_OUT_NORM,
};

struct t5_layer {
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    struct ggml_tensor * o;
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * wi_0;
    struct ggml_tensor * wi_1;
    struct ggml_tensor * wo;
    struct ggml_tensor * mlp_norm;
};

// this struct maintains the static tensors for a t5_encoder model
// the defautl configuration is form copied from standard configuration for
// flan-t5-xl. Note this model is slightly different from a standard t5 encoder.
// Specifically this model has a down projection which converts the text encoder's
// hidden size to the hidden size of the parler decoder.
struct t5_encoder : tts_model {
    // These configs  are essentially built for the 44khZ 8kbps standard DAC model audio encoder and decoder
    uint32_t n_layers = 24;
    uint32_t n_attn_heads = 32;
    uint32_t head_size = 64;
    uint32_t hidden_size = 2048;
    uint32_t relative_attn_buckets = 32;
    uint32_t eos_token_id = 1;
    uint32_t bos_token_id = 0;
    uint32_t max_context_length = 512;
    uint32_t output_size = 1536;
    uint32_t vocab_size;

    struct ggml_tensor * embd;
    struct ggml_tensor * relative_attn_bias;
    struct ggml_tensor * out_norm;
    struct ggml_tensor * down_proj = nullptr;
    struct ggml_tensor * down_proj_bias = nullptr;
    std::vector<t5_layer> layers;

    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "t5encoder", 1.25);
    }
};

// For assigning weights from gguf file to local model.
void assign_to_t5_encoder(t5_encoder * model, const std::string name, ggml_tensor * tensor);
void assign_to_t5_layer(t5_encoder * model, t5_layer & layer, std::string name, ggml_tensor * tensor);

struct t5_context : runner_context {
    t5_context(t5_encoder * model, int n_threads): runner_context(n_threads), model(model) {};

    struct t5_encoder * model;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * attn_mask;
    struct ggml_tensor * inp_pos_bucket;

    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }
};

struct t5_context * build_new_t5_context(struct t5_encoder * model, int n_threads, bool use_cpu = true);

struct t5_ubatch {
    size_t n_tokens; // the number of tokens in our encoded sequence
    uint32_t * input_tokens;    // [n_tokens]
};

static struct ggml_tensor * build_t5_norm(struct ggml_context * ctx, struct ggml_tensor * cur, struct ggml_tensor * weight);
static struct ggml_tensor * build_t5_attn_mask(ggml_context * ctx, struct t5_context *t5ctx, const t5_ubatch & batch);

// This struct is intended to manage the t5 encoder model's graph compilation and compute function.
struct t5_runner : tts_runner {
    t5_runner(t5_encoder * model, t5_context * context, unigram_tokenizer * tokenizer): model(model), t5ctx(context), tokenizer(tokenizer) {};
    ~t5_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete t5ctx;
    }
    struct unigram_tokenizer * tokenizer;
    t5_encoder * model;
    t5_context * t5ctx;

    void init_build() {
        tts_runner::init_build(&t5ctx->buf_compute_meta);
    }

    void prepare_post_load();
    struct t5_ubatch build_worst_case_batch();
    void set_inputs(t5_ubatch & batch);
    struct ggml_cgraph * build_t5_graph(t5_ubatch & batch);
    void run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs);
    int generate(std::string prompt, struct tts_response * response);
};

struct t5_runner * text_encoder_from_file(std::string file_path, int n_threads, unigram_tokenizer * tokenizer, bool cpu_only = true);

#endif

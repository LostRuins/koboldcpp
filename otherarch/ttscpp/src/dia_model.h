#pragma once

#include "dac_model.h"
#include "ttssampler.h"

struct dia_encoder_layer {
    struct ggml_tensor * k;
    struct ggml_tensor * q;
    struct ggml_tensor * v;
    struct ggml_tensor * o;
    struct ggml_tensor * self_attn_norm;

    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * out;
    struct ggml_tensor * mlp_norm;
};

struct dia_decoder_layer {
    struct ggml_tensor * self_attn_k;
    struct ggml_tensor * self_attn_q;
    struct ggml_tensor * self_attn_v;
    struct ggml_tensor * self_attn_o;
    struct ggml_tensor * self_attn_norm;

    struct ggml_tensor * cross_attn_k;
    struct ggml_tensor * cross_attn_q;
    struct ggml_tensor * cross_attn_v;
    struct ggml_tensor * cross_attn_o;
    struct ggml_tensor * cross_attn_norm;

    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * out;
    struct ggml_tensor * mlp_norm;

    struct ggml_tensor * pad_attn_values;
};

struct dia_encoder {
    struct ggml_tensor * norm;
    struct ggml_tensor * embedding;
    std::vector<dia_encoder_layer*> layers;
};

struct dia_decoder {
    struct ggml_tensor * norm;
    std::vector<struct ggml_tensor*> embds;
    std::vector<struct ggml_tensor*> heads;
    std::vector<dia_decoder_layer*> layers;
};

struct dia_model : tts_model {
    // These default configurations are based on the default configuration for the Dia 1.68b param model.
    uint32_t n_output_heads = 9;
    uint32_t n_encoder_layers = 12;
    uint32_t n_decoder_layers = 18;
    uint32_t encoder_hidden_size = 1024;
    uint32_t decoder_hidden_size = 2048;
    uint32_t encoder_attn_heads = 16;
    uint32_t decoder_attn_heads = 16;
    uint32_t decoder_query_heads = 4;
    uint32_t head_size = 128;
    uint32_t eos_token_id = 1024;
    uint32_t pad_token_id = 1025;
    uint32_t bos_token_id = 1026;
    uint32_t output_vocab_size = 1028;
    uint32_t audio_vocab_size = 1024;
    uint32_t max_generation_size = 3072;
    uint32_t max_encoder_context_length = 1024;


    float cfg_scale_data[2] = {3.0, 1024.0};
    uint32_t max_delay = 15;
    std::vector<uint32_t> delay_pattern = {0, 8, 9, 10, 11, 12, 13, 14, 15};

    dia_encoder * encoder;
    dia_decoder * decoder;

    void assign_weight(std::string name, ggml_tensor * tensor);
    void assign_to_encoder(std::vector<std::string> parts, struct ggml_tensor * tensor, std::string name);
    void assign_to_decoder(std::vector<std::string> parts, struct ggml_tensor * tensor, std::string name);
    void assign_to_encoder_layer(std::string part, dia_encoder_layer * layer, struct ggml_tensor * tensor);
    void assign_to_decoder_layer(std::string part, dia_decoder_layer * layer, struct ggml_tensor * tensor);
    void prep_constants(gguf_context * meta);
    void prep_layers();
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers();
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "dia", 1.30);
    }
};

struct dia_context : runner_context {
    dia_context(dia_model * model, int n_threads): runner_context(n_threads), model(model) {
        max_generation_size = model->max_generation_size;
    };

    uint32_t current_position = 0;  // current position in the active sequence
    int delay_steps           = -1; // the max remaining steps to take before terminating; is set after an eos token is seen on the first output channel
    size_t prompt_size        = 0;

    uint32_t max_generation_size; // this is set by the generation context or defaults to the config set on dia model.

    std::vector<uint32_t> output_tokens;
    struct dia_model * model;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * audio_inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * encode_positions;
    struct ggml_tensor * encode_attn_mask;
    struct ggml_tensor * cross_attn_mask;

    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }
    void reset();
};

struct dia_kv_cache {
    ggml_type tensor_type = GGML_TYPE_F32;

    std::vector<struct ggml_tensor *> cross_k_l;
    std::vector<struct ggml_tensor *> cross_v_l;

    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;

    struct ggml_context * ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t buf;

    void free() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }

    ~dia_kv_cache() {
        free();
    }
};

struct dia_ubatch {
    dia_ubatch(size_t sequence_length, bool encoder_step = false): sequence_length(sequence_length), encoder_step(encoder_step) {};
    bool encoder_step; // whether we are performing the prompt encoding in this step.
    size_t sequence_length; // for just audio tokens the sequence length should be the total_tokens / num_heads; for normal generation this should always be 1.
    size_t sentence_length; // the number of non padded tokens in the conditional context
    std::vector<uint32_t> tokens; // character tokens for the encoder
    std::vector<uint32_t> audio_tokens; // audio tokens from the last generation
};

struct dia_context * build_new_dia_context(struct dia_model * model, int n_threads, bool use_cpu = true);
static bool dia_kv_cache_init(struct dia_kv_cache * cache, dia_model * model, dia_context * dctx) ;
static struct ggml_tensor * build_dia_decoder_inp_embd(struct ggml_context * ctx, dia_context *dctx, dia_decoder * decoder, dia_ubatch & batch, uint32_t n_output_heads);
static struct ggml_tensor * dia_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight);
static struct ggml_tensor * build_dia_encoder_attn_mask(ggml_context * ctx, struct dia_context * dctx, dia_model * model);
static struct ggml_tensor * build_dia_decoder_attn_mask(ggml_context * ctx, struct dia_context * dctx, dia_ubatch & batch);
static struct ggml_tensor * build_dia_decoder_cross_attn_mask(ggml_context * ctx, struct dia_context * dctx, dia_ubatch & batch);
static struct ggml_tensor * build_dia_head_outputs(struct ggml_context * ctx, dia_model * model, struct ggml_tensor * cur);
static struct ggml_tensor * build_dia_encoder(ggml_context * ctx, dia_model * model, dia_context * dctx, dia_ubatch & batch);
static void build_dia_self_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv, ggml_cgraph * gf, struct ggml_tensor * k, struct ggml_tensor * v, dia_ubatch & batch, int layer_index);
static void build_dia_cross_kv_store(ggml_context * ctx, dia_context * dctx, dia_model * model, dia_kv_cache * kv, ggml_cgraph * gf, struct ggml_tensor * encoder_hidden_states, int layer_index);
static struct ggml_tensor * build_dia_decoder( ggml_cgraph * gf, ggml_context * ctx, dia_model * model,  dia_context * dctx,  dia_kv_cache * cache, dia_ubatch & batch, struct ggml_tensor * encoder_hidden_states);

// This struct is intended to support end-to-end TTS generation for the Dia model. As such, it manages Dia's model compilation, compute, generation,
// tokenizationm and sampling process, and uses the dac_runner struct to encode audio outputs.
struct dia_runner : tts_runner {
    dia_runner(dia_model * model, dac_runner * audio_decoder, dia_context * dctx, sampler * samp, dia_kv_cache * cache): model(model), dac_runner(audio_decoder), dctx(dctx), decode_sampler(samp), kv_cross_self(cache) {
        decode_sampler->vocab_size = model->output_vocab_size;
    };
    ~dia_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete kv_cross_self;
        delete dac_runner;
        delete dctx;
        delete decode_sampler;
    }
    struct dia_model * model;
    struct dac_runner * dac_runner;
    struct dia_context * dctx;
    struct dia_kv_cache * kv_cross_self = nullptr;
    struct sampler * decode_sampler;

    void init_build() {
        tts_runner::init_build(&dctx->buf_compute_meta);
    }

    void tokenize_sentence(std::string sentence, dia_ubatch & tokens);
    dia_ubatch batch_from_sentence(std::string sentence);
    void configure_generation(generation_configuration * config);
    void assign_weight(std::string name, ggml_tensor * tensor);
    dia_ubatch build_worst_case_batch();
    struct ggml_cgraph * build_dia_graph(dia_ubatch & batch);
    void set_inputs(dia_ubatch & batch);
    int decode(dia_ubatch & batch);
    void prepare_post_load();
    int generate(std::string sentence, struct tts_response * response);
    bool check_stopping(dia_ubatch & batch);
    void adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered);
    int generate_from_batch(dia_ubatch & batch, struct tts_response * output);
};

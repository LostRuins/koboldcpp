#ifndef parler_model_h
#define parler_model_h

#include "dac_model.h"
#include "ttst5_encoder_model.h"
#include "ttssampler.h"

enum parler_tensor {
    PARLER_EMBD,
    PARLER_EMBD_PROMPTS,
    PARLER_TEXT_ENCODING,
    PARLER_POSITIONAL_EMBD,
    PARLER_HEAD,
    PARLER_NORM,
    PARLER_NORM_BIAS,
    PARLER_LAYER_SELF_ATTN_Q,
    PARLER_LAYER_SELF_ATTN_K,
    PARLER_LAYER_SELF_ATTN_V,
    PARLER_LAYER_SELF_ATTN_O,
    PARLER_LAYER_SELF_ATTN_NORM,
    PARLER_LAYER_SELF_ATTN_NORM_BIAS,
    PARLER_LAYER_ATTN_Q,
    PARLER_LAYER_ATTN_K,
    PARLER_LAYER_ATTN_V,
    PARLER_LAYER_ATTN_O,
    PARLER_LAYER_ATTN_NORM,
    PARLER_LAYER_ATTN_NORM_BIAS,
    PARLER_LAYER_FC1,
    PARLER_LAYER_FC2,
    PARLER_LAYER_OUT_NORM,
    PARLER_LAYER_OUT_NORM_BIAS,
};

struct parler_layer {
    struct ggml_tensor * self_attn_k_proj;
    struct ggml_tensor * self_attn_q_proj;
    struct ggml_tensor * self_attn_v_proj;
    struct ggml_tensor * self_attn_o_proj;
    struct ggml_tensor * self_attn_norm;
    struct ggml_tensor * self_attn_norm_bias;

    struct ggml_tensor * attn_k_proj;
    struct ggml_tensor * attn_q_proj;
    struct ggml_tensor * attn_v_proj;
    struct ggml_tensor * attn_o_proj;
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_bias;

    struct ggml_tensor * cross_k;
    struct ggml_tensor * cross_v;

    struct ggml_tensor * fc1;
    struct ggml_tensor * fc2;
    struct ggml_tensor * final_norm;
    struct ggml_tensor * final_norm_bias;
};

struct parler_tts_model : tts_model {
    // These default configurations are based on the configuration of Parler TTS Mini (version 1.0)
    uint32_t n_output_heads = 9;
    uint32_t n_encode_length;
    uint32_t max_encode_length = 512; // This corresponds with the max token length of the conditional prompt
    uint32_t hidden_size = 1024;
    uint32_t max_ctx_length = 4096;
    uint32_t n_attn_heads = 16;
    uint32_t head_size = 64;
    uint32_t output_vocab_size = 1088;
    uint32_t eos_token_id = 1024;
    uint32_t audio_vocab_size = 1024;
    uint32_t max_generation_size = 2580;
    uint32_t n_layers = 24;
    uint32_t bos_token_id = 1025;
    uint32_t max_cross_nodes = 32;
    uint32_t prompt_vocab_size;

    bool use_cross_attn = true;

    std::vector<struct ggml_tensor*> embds;
    std::vector<parler_layer*> layers;
    std::vector<struct ggml_tensor*> heads;

    struct ggml_tensor * precomputed_input_emb;
    struct ggml_tensor * precomputed_positional_embds;

    struct ggml_tensor * layer_norm;
    struct ggml_tensor * layer_norm_bias;
    struct ggml_tensor * prompt_embd;

    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void prep_cross_key_values(int n_threads, struct tts_response * conditional_prompt = nullptr);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "decoder", 1.30, max_encode_length*hidden_size*sizeof(float)*n_layers*2);
    }
};

// For assigning weights to the parler model from a gguf file.
void assign_parler_layer(parler_tts_model * model, parler_layer & layer, std::string name, ggml_tensor * tensor);
void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor);

struct parler_context : runner_context {
    parler_context(parler_tts_model * model, int n_threads): runner_context(n_threads), model(model) {};
    struct parler_tts_model * model;
    std::vector<bool> eos_seen;

    bool use_cache = true;

    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch
    uint32_t current_position = 0; // current position in the active sequence
    uint32_t prompt_end_position = 0; // the position of the text prompt termination (used for adjusting the cache when incrementally generating)
    int32_t seq_id; // a unique identifier associated with the active sequence.

    std::vector<uint32_t> output_tokens;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * audio_inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * attn_mask;
    struct ggml_tensor * attn_mask_cross;

    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }
    void reset(int32_t n_output_heads);
};

struct parler_kv_cache {
    int32_t seq_id;

    ggml_type type_k = GGML_TYPE_F32;
    ggml_type type_v = GGML_TYPE_F32;

    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;

    struct ggml_context * ctx;
    ggml_backend_buffer_type_t buft;
    ggml_backend_buffer_t buf;

    void free() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
    }

    ~parler_kv_cache() {
        free();
    }
};

struct parler_ubatch {
    parler_ubatch(bool audio_generation, size_t n_tokens, size_t n_audio_tokens, size_t sequence_length,
        uint32_t * tokens, uint32_t * audio_tokens, uint32_t * positions, uint32_t * true_order,
        int current_step): audio_generation(audio_generation), n_tokens(n_tokens), n_audio_tokens(n_audio_tokens), sequence_length(sequence_length), tokens(tokens), audio_tokens(audio_tokens), positions(positions), true_order(true_order), current_step(current_step) {};
    parler_ubatch() {};
    bool audio_generation; // whether we are receiving codebook decoded tokens or text tokens
    size_t n_tokens; // total sentence tokens
    size_t n_audio_tokens; // total audio tokens
    size_t sequence_length; // for just audio tokens the sequence length should be the total_tokens / num_heads; in general this should be n_tokens + n_audio_tokens / num_heads
    uint32_t * tokens;    // [n_tokens]
    uint32_t * audio_tokens; // [n_audio_tokens]
    uint32_t * positions; // [sequence_length]
    uint32_t * true_order;
    int current_step = 0; // total_generations
};

struct parler_context * build_new_parler_context(struct parler_tts_model * model, int n_threads, bool use_cpu = true);
static bool parler_kv_cache_init(struct parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx, int32_t seq_id);

struct ggml_tensor * parler_build_inp_embd(struct ggml_context * ctx, struct parler_context * pctx, parler_tts_model * model, const parler_ubatch & batch);
struct ggml_tensor * parler_build_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight, struct ggml_tensor * bias);
void parler_build_kv_store(struct ggml_context * ctx, const parler_kv_cache * kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa);
struct ggml_tensor * parler_build_head_outputs(struct ggml_context * ctx, parler_tts_model * model, struct ggml_tensor * cur);
struct ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch);
struct ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model, parler_ubatch & batch);
static struct parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer);

// This struct is intended to support end-to-end TTS generation. As such, it manages the parler tts model compilation, compute and generation process,
// the tokenization and sampling process, and uses the dac_runner struct to encode audio outputs.
struct parler_tts_runner : tts_runner {
    parler_tts_runner(parler_tts_model * model, dac_runner * audio_decoder, parler_context * pctx, unigram_tokenizer * ut, sampler * samp, parler_kv_cache * cache): model(model), dac_runner(audio_decoder), pctx(pctx), tokenizer(ut), sampler(samp), kv_self(cache) {};
    ~parler_tts_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete kv_self;
        delete dac_runner;
        delete pctx;
        delete sampler;
    }
    struct parler_tts_model * model;
    struct dac_runner * dac_runner;
    struct parler_context * pctx;
    struct unigram_tokenizer * tokenizer;
    struct parler_kv_cache * kv_self = nullptr;
    struct sampler * sampler;

    void init_build() {
        tts_runner::init_build(&pctx->buf_compute_meta);
    }

    void configure_generation(generation_configuration * config);
    void assign_weight(std::string name, ggml_tensor * tensor);
    parler_ubatch build_worst_case_batch();
    struct ggml_cgraph * build_parler_graph(parler_ubatch & batch);
    void set_inputs(parler_ubatch & batch);
    int decode(parler_ubatch & batch);
    void prepare_post_load();
    bool adjust_for_sequence_continuation(struct parler_ubatch & batch);
    int generate(std::string sentence, struct tts_response * response, int32_t seq_id = -1);
    bool check_stopping();
    void adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered);
    int generate_from_batch(parler_ubatch & batch, struct tts_response * output);
    void parler_graph_compute(ggml_cgraph * gf);
    void just_audio_token_decode(uint32_t * tokens, int32_t sq_len, struct tts_response * output);
    int generate_audio_tokens(std::string sentence);
    void update_conditional_prompt(const std::string file_path, const std::string prompt, int n_threads, bool cpu_only = true);
};

#endif

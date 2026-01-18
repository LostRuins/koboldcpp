#ifndef kokoro_model_h
#define kokoro_model_h

#include <stdlib.h>
#include "tts_model.h"
#include "ttstokenizer.h"
#include "phonemizer.h"

// Rather than using ISO 639-2 language codes, Kokoro voice pack specify their corresponding language via their first letter.
// Below is a map that describes the relationship between those designations and espeak-ng's voice identifiers so that the
// appropriate phonemization protocol can inferred from the Kokoro voice.
static std::map<char, std::string> KOKORO_LANG_TO_ESPEAK_ID = {
	{'a', "gmw/en-US"},
	{'b', "gmw/en"},
	{'e', "roa/es"},
	{'f', "roa/fr"},
	{'h', "inc/hi"},
	{'i', "roa/it"},
	{'j', "jpx/ja"},
	{'p', "roa/pt-BR"},
	{'z', "sit/cmn"}
};

struct lstm_cell {
	std::vector<ggml_tensor*> weights;
	std::vector<ggml_tensor*> biases;
	std::vector<ggml_tensor*> reverse_weights;
	std::vector<ggml_tensor*> reverse_biases;
};

struct lstm {
	std::vector<ggml_tensor*> hidden;
	std::vector<ggml_tensor*> states;

	bool bidirectional = false;
	std::vector<lstm_cell*> cells;
};

struct duration_predictor_layer {
	lstm * rnn;
	struct ggml_tensor * ada_norm_gamma_weight;
	struct ggml_tensor * ada_norm_gamma_bias;
	struct ggml_tensor * ada_norm_beta_weight;
	struct ggml_tensor * ada_norm_beta_bias;
};

struct ada_residual_conv_block {
	struct ggml_tensor * conv1;
	struct ggml_tensor * conv1_bias;
	struct ggml_tensor * conv2;
	struct ggml_tensor * conv2_bias;
	struct ggml_tensor * norm1_gamma;
	struct ggml_tensor * norm1_gamma_bias;
	struct ggml_tensor * norm1_beta;
	struct ggml_tensor * norm1_beta_bias;
	struct ggml_tensor * norm2_gamma;
	struct ggml_tensor * norm2_gamma_bias;
	struct ggml_tensor * norm2_beta;
	struct ggml_tensor * norm2_beta_bias;
	struct ggml_tensor * pool = nullptr;
	struct ggml_tensor * pool_bias = nullptr;
	struct ggml_tensor * upsample = nullptr;
	struct ggml_tensor * upsample_bias = nullptr;
};

struct duration_predictor {
	struct ggml_tensor * albert_encode;
	struct ggml_tensor * albert_encode_bias;
	std::vector<duration_predictor_layer*> layers;
	lstm * duration_proj_lstm;
	struct ggml_tensor * duration_proj;
	struct ggml_tensor * duration_proj_bias;
	struct ggml_tensor * n_proj_kernel;
	struct ggml_tensor * n_proj_bias;
	struct ggml_tensor * f0_proj_kernel;
	struct ggml_tensor * f0_proj_bias;
	lstm * shared_lstm;
	std::vector<ada_residual_conv_block*> f0_blocks;
	std::vector<ada_residual_conv_block*> n_blocks;
};

struct kokoro_text_encoder_conv_layer {
	struct ggml_tensor * norm_gamma;
	struct ggml_tensor * norm_beta;
	struct ggml_tensor * conv_weight;
	struct ggml_tensor * conv_bias;
};

struct kokoro_text_encoder {
	struct ggml_tensor * embd;
	std::vector<kokoro_text_encoder_conv_layer*> conv_layers;
	lstm * out_lstm;
};

struct kokoro_generator_residual_block {
	std::vector<uint32_t> conv1_dilations;
	std::vector<uint32_t> conv1_paddings;

	std::vector<ggml_tensor*> adain1d_1_gamma_weights;
	std::vector<ggml_tensor*> adain1d_2_gamma_weights;
	std::vector<ggml_tensor*> adain1d_1_gamma_biases;
	std::vector<ggml_tensor*> adain1d_2_gamma_biases;
	std::vector<ggml_tensor*> adain1d_1_beta_weights;
	std::vector<ggml_tensor*> adain1d_2_beta_weights;
	std::vector<ggml_tensor*> adain1d_1_beta_biases;
	std::vector<ggml_tensor*> adain1d_2_beta_biases;
	std::vector<ggml_tensor*> input_alphas;
	std::vector<ggml_tensor*> output_alphas;
	std::vector<ggml_tensor*> convs1_weights;
	std::vector<ggml_tensor*> convs1_biases;
	std::vector<ggml_tensor*> convs2_weights;
	std::vector<ggml_tensor*> convs2_biases;
};

struct kokoro_noise_residual_block {
	uint32_t input_conv_stride;
	uint32_t input_conv_padding;

	struct ggml_tensor * input_conv;
	struct ggml_tensor * input_conv_bias;
	struct kokoro_generator_residual_block * res_block;
};

struct kokoro_generator_upsample_block {
	uint32_t padding;
	uint32_t stride;

	// these are just conv transpose layers
	struct ggml_tensor * upsample_weight;
	struct ggml_tensor * upsample_bias;
};

struct kokoro_generator {
	// unfortunately the squared sum of the windows needs to be computed dynamically per run because it is dependent
	// on the sequence size of the generation and the hop is typically less than half the size of our window.
	struct ggml_tensor * window;

	struct ggml_tensor * m_source_weight;
	struct ggml_tensor * m_source_bias;
	struct ggml_tensor * out_conv_weight;
	struct ggml_tensor * out_conv_bias;
	std::vector<kokoro_noise_residual_block*> noise_blocks;
	std::vector<kokoro_generator_residual_block*> res_blocks;
	std::vector<kokoro_generator_upsample_block*> ups;
};

struct kokoro_decoder {
	struct ggml_tensor * f0_conv;
	struct ggml_tensor * f0_conv_bias;
	struct ggml_tensor * n_conv;
	struct ggml_tensor * n_conv_bias;
	struct ggml_tensor * asr_conv;
	struct ggml_tensor * asr_conv_bias;
	std::vector<ada_residual_conv_block*> decoder_blocks;
	ada_residual_conv_block* encoder_block;
	kokoro_generator * generator;
};

struct albert_layer {
	struct ggml_tensor * ffn;
	struct ggml_tensor * ffn_out;
	struct ggml_tensor * ffn_bias;
	struct ggml_tensor * ffn_out_bias;
	struct ggml_tensor * layer_output_norm_weight;
	struct ggml_tensor * layer_output_norm_bias;
	struct ggml_tensor * q;
	struct ggml_tensor * k;
	struct ggml_tensor * v;
	struct ggml_tensor * o;
	struct ggml_tensor * q_bias;
	struct ggml_tensor * k_bias;
	struct ggml_tensor * v_bias;
	struct ggml_tensor * o_bias;
	struct ggml_tensor * attn_norm_weight;
	struct ggml_tensor * attn_norm_bias;
};

struct kokoro_model : tts_model {
	// standard configruation for Kokoro's Albert model
	// tokenization
	uint32_t bos_token_id = 0;
	uint32_t eos_token_id = 0;
	uint32_t space_token_id = 16;
	// duration prediction
	uint32_t max_context_length = 512;
	uint32_t vocab_size = 178;
	uint32_t hidden_size = 768;
	uint32_t n_attn_heads = 12;
	uint32_t n_layers = 1;
	uint32_t n_recurrence = 12;
	uint32_t head_size = 64;
	uint32_t duration_hidden_size = 512;
	uint32_t up_sampling_factor;
	float upsample_scale = 300.0f;
	float scale = 0.125f;

	// standard configuration for duration prediction
	uint32_t f0_n_blocks = 3;
	uint32_t n_duration_prediction_layers = 3;
	// while it is technically possible for the duration predictor to assign 50 values per token there is no practical need to
	// allocate that many items to the sequence as it is impossible for all tokens to require such long durations and each
	// allocation increases node allocation size by O(N)
	uint32_t max_duration_per_token = 20;
	uint32_t style_half_size = 128;

	// standard text encoding configuration
	uint32_t n_conv_layers = 3;

	// standard decoder configuration
	uint32_t n_kernels = 3;
	uint32_t n_upsamples = 2;
	uint32_t n_decoder_blocks = 4;
	uint32_t n_res_blocks = 6;
	uint32_t n_noise_blocks = 2;
	uint32_t out_conv_padding = 3;
	uint32_t post_n_fft = 11;
	uint32_t true_n_fft = 20;
	uint32_t stft_hop = 5;
	uint32_t harmonic_num = 8;
	float sin_amp = 0.1f;
	float noise_std = 0.003f;
	float voice_threshold = 10.0f;
	float sample_rate = 24000.0f;
	std::string window = "hann";

	// It is really annoying that ggml doesn't allow using non ggml tensors as the operator for simple math ops.
	// This is just the constant defined above as a tensor.
	struct ggml_tensor * n_kernels_tensor;

	// Kokoro loads albert with use_pooling = true but doesn't use the pooling outputs.
	bool uses_pooling = false;
	bool static_token_types = true;

	std::map<std::string, struct ggml_tensor *> voices;

	// Albert portion of the model
	struct ggml_tensor * embd_hidden;
	struct ggml_tensor * embd_hidden_bias;
	struct ggml_tensor * token_type_embd = nullptr;
	struct ggml_tensor * token_embd;
	struct ggml_tensor * position_embd;
	struct ggml_tensor * input_norm_weight;
	struct ggml_tensor * input_norm_bias;
	struct ggml_tensor * static_token_type_values = nullptr;
	struct ggml_tensor * pool = nullptr;
	struct ggml_tensor * pool_bias = nullptr;
	std::vector<albert_layer*> layers;

	struct ggml_tensor * harmonic_sampling_norm = nullptr; // a static 1x9 harmonic multiplier
	struct ggml_tensor * sampling_factor_scalar = nullptr; // a static scalar
	struct ggml_tensor * sqrt_tensor = nullptr; // static tensor for constant division

	// Prosody Predictor portion of the model
	struct duration_predictor * prosody_pred;

	// Text encoding portion of the model
	struct kokoro_text_encoder * text_encoder;

	// Decoding and Generation portion of the model
	struct kokoro_decoder * decoder;

	// the default hidden states need to be initialized
	std::vector<lstm*> lstms;

	size_t duration_node_counter = 0;
	size_t generation_node_counter = 0;
	// setting this is likely unnecessary as it is precomputed by the post load function.
	uint32_t post_load_tensor_bytes = 13000;

	size_t max_gen_nodes();
	size_t max_duration_nodes();

	lstm * prep_lstm();
	// helper functions for assigning tensors to substructs
	void assign_lstm(lstm * rnn, std::string name, ggml_tensor * tensor);
	void assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor);
	void assign_gen_resblock(kokoro_generator_residual_block * block, std::string name, ggml_tensor * tensor);
	void assign_ada_res_block(ada_residual_conv_block * block, std::string name, ggml_tensor * tensor);
	void assign_decoder_weight(std::string name, ggml_tensor * tensor);
	void assign_duration_weight(std::string name, ggml_tensor * tensor);
	void assign_text_encoder_weight(std::string name, ggml_tensor * tensor);
	void assign_albert_weight(std::string name, ggml_tensor * tensor);


	void post_load_assign();
    void assign_weight(std::string name, ggml_tensor * tensor);
    void prep_layers(gguf_context * meta);
    void prep_constants(gguf_context * meta);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
    	std::function<void (ggml_tensor *)> fn = ([&](ggml_tensor* cur) {
    		std::string name = ggml_get_name(cur);
    		size_t increment = 1;
    		if (name.find("lstm") != std::string::npos) {
    			increment = max_context_length;
    		}
    		if (name.find("duration_predictor") != std::string::npos) {
    			duration_node_counter += increment;
    		} else {
    			generation_node_counter += increment;
    		}
    	});
    	compute_tensor_meta_cb = &fn;
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "kokoro", 1.6, post_load_tensor_bytes);
    }
};

struct kokoro_ubatch {
    size_t n_tokens; // the number of tokens in our encoded sequence
    uint32_t * input_tokens;    // [n_tokens]
    struct kokoro_duration_response * resp = nullptr;
};

struct kokoro_duration_context : runner_context {
    kokoro_duration_context(kokoro_model * model, int n_threads): runner_context(n_threads), model(model) {};
    ~kokoro_duration_context() {
        ggml_backend_buffer_free(buf_len_output);
    }

    std::string voice = "af_heart";
    struct kokoro_model * model;
    ggml_backend_buffer_t buf_len_output = nullptr;


    size_t  logits_size = 0; // capacity (of floats) for logits
    float * lens 		= nullptr;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * positions;
    struct ggml_tensor * attn_mask;
    struct ggml_tensor * token_types = nullptr;

    void build_schedule() {
        runner_context::build_schedule(model->max_duration_nodes()*5);
    }
};

static struct ggml_tensor * build_albert_attn_mask(ggml_context * ctx, struct kokoro_duration_context *kctx, const kokoro_ubatch & batch);
static struct ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types);
static struct ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias);
static struct ggml_tensor * build_ada_residual_conv(ggml_context * ctx, struct ggml_tensor * x, ada_residual_conv_block * block, struct ggml_tensor * style, struct ggml_tensor * sqrt_tensor);
static struct ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * style, kokoro_generator_residual_block * block);
static struct ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, struct ggml_tensor * x, struct ggml_tensor * style);
static kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key);
static kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index);
static kokoro_generator_upsample_block* kokoro_generator_upsample_block(gguf_context * meta, int index);

std::string get_espeak_id_from_kokoro_voice(std::string voice);
struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

struct kokoro_duration_response {
	size_t n_outputs;
	float * lengths;
	float * hidden_states;
};

// This struct is intended to manage graph and compute for the duration prediction portion of the kokoro model.
// Duration computation and speech generation are separated into distinct graphs because the precomputed graph structure of ggml doesn't
// support the tensor dependent views that would otherwise be necessary.
struct kokoro_duration_runner : tts_runner {
    kokoro_duration_runner(kokoro_model * model, kokoro_duration_context * context, single_pass_tokenizer * tokenizer): model(model), kctx(context), tokenizer(tokenizer) {};
    ~kokoro_duration_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete kctx;
    }
    struct single_pass_tokenizer * tokenizer;
    kokoro_model * model;
    kokoro_duration_context * kctx;

    void init_build() {
        tts_runner::init_build(&kctx->buf_compute_meta);
    }

    void prepare_post_load();
    struct kokoro_ubatch build_worst_case_batch();
    void set_inputs(kokoro_ubatch & batch);
    struct ggml_cgraph * build_kokoro_duration_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & ubatch);
};

struct kokoro_context : runner_context {
    kokoro_context(kokoro_model * model, int n_threads): runner_context(n_threads), model(model) {};
    ~kokoro_context() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cpu);
        if (backend) {
            ggml_backend_free(backend);
        }
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
        }
    }

    std::string voice = "af_heart";

    struct kokoro_model * model;

    uint32_t total_duration;
    uint32_t sequence_length;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * duration_pred;
    struct ggml_tensor * duration_mask;
    struct ggml_tensor * window_sq_sum; // needs to be calculatd from the generator window.
    struct ggml_tensor * uv_noise_data;

    void build_schedule() {
        runner_context::build_schedule(model->max_gen_nodes()*30);
    }
};

// TODO: now that we are passing the context down to these methods we should clean up their parameters
static struct ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, struct ggml_tensor * style, struct ggml_tensor * f0_curve, kokoro_generator* generator, int sequence_length, struct ggml_tensor * window_sq_sum, ggml_cgraph * gf);
static struct ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std);

struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu = true);

// This manages the graph compilation of computation for the Kokoro model.
struct kokoro_runner : tts_runner {
    kokoro_runner(kokoro_model * model, kokoro_context * context, single_pass_tokenizer * tokenizer, kokoro_duration_runner * drunner, phonemizer * phmzr): model(model), kctx(context), tokenizer(tokenizer), drunner(drunner), phmzr(phmzr) {
    	tts_runner::sampling_rate = 24000.0f;
    	tts_runner::supports_voices = true;
    };
    ~kokoro_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        delete drunner;
        model->free();
        delete model;
        delete kctx;
        delete phmzr;
    }
    struct single_pass_tokenizer * tokenizer;
    kokoro_model * model;
    kokoro_context * kctx;
    kokoro_duration_runner * drunner;
    phonemizer * phmzr;

    std::string default_voice = "af_heart";

    void init_build() {
        tts_runner::init_build(&kctx->buf_compute_meta);
    }

    std::vector<std::string> list_voices();
    std::vector<std::vector<uint32_t>> tokenize_chunks(std::vector<std::string> clauses);
    void assign_weight(std::string name, ggml_tensor * tensor);
    void prepare_post_load();
    kokoro_ubatch build_worst_case_batch();
    void set_inputs(kokoro_ubatch & batch, uint32_t total_size);
    struct ggml_cgraph * build_kokoro_graph(kokoro_ubatch & batch);
    void run(kokoro_ubatch & batch, struct tts_response * outputs);
    int generate(std::string prompt, struct tts_response * response, std::string voice, std::string voice_code = "");
};

#endif

#pragma once

#include "general_neural_audio_codec.h"

// SNAC, Scale Neural Audio Codec, is another neural audio codec much like DAC.
// The key differences are that it uses grouping in the residual units of its layers,
// performs a repeat_interleave over the second and third input channels, applies
// a noise convolutional layer after input encoding for each layer, and applies
// an extra convolutional layer before residual layers are applied.
struct snac_model : tts_model {
    // general configuration from SNAC as used by Orpheus
    uint32_t n_layers = 4;
    uint32_t n_heads = 3;
    uint32_t up_sampling_factor = 512;
    uint32_t embd = 768;
    size_t max_generation_size = 2580;
    uint32_t repeats[3] = {4, 2, 1};
    // configuration for adding noise
    uint32_t noise_steps[4] = {8, 64, 256, 512};
    uint32_t noise_steps_sum = 840;
    bool use_noise = true;

    struct ggml_tensor * repeat_interleave_buffer;

    struct ggml_tensor * in_conv_kernel;
    struct ggml_tensor * in_conv_bias;
    struct ggml_tensor * up_conv_kernel;
    struct ggml_tensor * up_conv_bias;
    struct ggml_tensor * out_conv_kernel;
    struct ggml_tensor * out_conv_bias;
    struct ggml_tensor * snake_alpha;
    std::vector<general_neural_audio_codec::layer> layers;
    std::vector<general_neural_audio_codec::residual_vector_quantize_layer> quantizer_layers;

    void assign_weight(std::string name, ggml_tensor * weight);
    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void post_load_assign();
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
        prep_layers(meta_ctx);
        prep_constants(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "snac");
    }
};

// the context used for running the snac model
struct snac_context : runner_context {
    snac_context(snac_model * model, int n_threads): runner_context(n_threads), model(model) {};

    struct snac_model * model;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * noise;

    void build_schedule() {
        runner_context::build_schedule(model->max_nodes());
    }
};

snac_context * build_new_snac_context(struct snac_model * model, int n_threads, bool use_cpu = true);

static struct ggml_tensor * snac_build_audio_inputs(struct ggml_context * ctx, struct snac_context * sctx, size_t sequence_length, std::vector<general_neural_audio_codec::residual_vector_quantize_layer> layers);

// This struct is intended to manage the snac model's graph compilation and compute function.
struct snac_runner : tts_runner {
    snac_runner(snac_model * model, snac_context * context): model(model), sctx(context) {};
    ~snac_runner() {
        if (ctx) {
            ggml_free(ctx);
        }
        model->free();
        delete model;
        delete sctx;
    }
    snac_model * model;
    snac_context * sctx;

    void init_build() {
        tts_runner::init_build(&sctx->buf_compute_meta);
    }

    void set_inputs(std::vector<std::vector<uint32_t>> & tokens);
    void prepare_post_load();
    struct ggml_cgraph * build_snac_graph(size_t sequence_length);
    void run(std::vector<std::vector<uint32_t>> & tokens, struct tts_response * outputs);
};

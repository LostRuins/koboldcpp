// tokenizer.h: FSQ + Detokenizer via ggml
//
// Decodes LM audio codes into context_latents for DiT:
//   codes [T_5Hz] -> FSQ decode -> [T_5Hz, 6] -> project_out -> [T_5Hz, 2048]
//   -> detokenizer (per token): embed + special_tokens broadcast + 2L encoder + proj_out
//   -> [T_25Hz, 64] context_latents (T_25Hz = T_5Hz * 5)
//
// Weights live in the DiT GGUF (prefix "tokenizer." and "detokenizer.")
// Detokenizer reuses Qwen3 encoder infrastructure from qwen3.h

#pragma once
#include "qwen3.h"

// FSQ constants
static const int FSQ_NDIMS = 6;
static const int FSQ_LEVELS[6] = {8, 8, 8, 5, 5, 5};

// FSQ decode: integer index -> 6 normalized float values
// Each dimension: level_idx / ((L-1)/2) - 1.0  (maps to [-1, 1])
static void fsq_decode_index(int index, float * out) {
    int stride = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        int L = FSQ_LEVELS[d];
        int level_idx = (index / stride) % L;
        float half_L = (float)(L - 1) / 2.0f;
        out[d] = (float)level_idx / half_L - 1.0f;
        stride *= L;
    }
}

// Detokenizer config: same Qwen3 arch as lyric/timbre encoders, 2 layers
static Qwen3Config detok_config() {
    return {
        /*hidden_size*/       2048,
        /*intermediate_size*/ 6144,
        /*n_heads*/           16,
        /*n_kv_heads*/        8,
        /*head_dim*/          128,
        /*n_layers*/          2,
        /*rope_theta*/        1000000.0f,
        /*rms_norm_eps*/      1e-6f,
        /*is_causal*/         false,
    };
}

struct DetokGGML {
    // FSQ project_out: Linear(6, 2048) + bias
    struct ggml_tensor * fsq_proj_w;    // [2048, 6]
    struct ggml_tensor * fsq_proj_b;    // [2048]

    // Detokenizer
    struct ggml_tensor * embed_w;       // [2048, 2048]
    struct ggml_tensor * embed_b;       // [2048]
    struct ggml_tensor * special_tok;   // [2048, 5] (broadcast positional)
    Qwen3Config cfg;
    Qwen3Layer layers[2];
    struct ggml_tensor * norm;          // [2048]
    struct ggml_tensor * proj_out_w;    // [64, 2048]
    struct ggml_tensor * proj_out_b;    // [64]

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx wctx;
};

// Load from DiT GGUF
static bool detok_ggml_load(DetokGGML * m, const char * gguf_path,
                             ggml_backend_t backend, ggml_backend_t cpu_backend) {
    m->cfg = detok_config();
    m->backend = backend;
    m->cpu_backend = cpu_backend;

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // FSQ(2) + embed(2) + special(1) + 2 layers x 11(22) + norm(1) + proj(2) = 30
    wctx_init(&m->wctx, 30);

    m->fsq_proj_w = gf_load_tensor(&m->wctx, gf, "tokenizer.quantizer.project_out.weight");
    m->fsq_proj_b = gf_load_tensor(&m->wctx, gf, "tokenizer.quantizer.project_out.bias");

    m->embed_w = gf_load_tensor(&m->wctx, gf, "detokenizer.embed_tokens.weight");
    m->embed_b = gf_load_tensor(&m->wctx, gf, "detokenizer.embed_tokens.bias");
    m->norm    = gf_load_tensor(&m->wctx, gf, "detokenizer.norm.weight");
    m->proj_out_w = gf_load_tensor(&m->wctx, gf, "detokenizer.proj_out.weight");
    m->proj_out_b = gf_load_tensor(&m->wctx, gf, "detokenizer.proj_out.bias");

    // special_tokens: GGUF [2048, 5, 1] (ggml order), reshape to [2048, 5]
    m->special_tok = gf_load_tensor(&m->wctx, gf, "detokenizer.special_tokens");

    for (int i = 0; i < m->cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "detokenizer.layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->layers[i], prefix);
    }

    if (!wctx_alloc(&m->wctx, backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    // Scheduler
    ggml_backend_t backends[2] = { backend, cpu_backend };
    int n = (backend == cpu_backend) ? 1 : 2;
    m->sched = ggml_backend_sched_new(backends, NULL, n, 4096, false, true);

    fprintf(stderr, "[Load] Detokenizer: FSQ(6->2048) + 2L encoder(S=5, 2048->64)\n");
    return true;
}

// Decode LM audio codes -> context_latents
// codes: [T_5Hz] integer array
// context_out: [64 * T_25Hz] flat, caller allocates (T_25Hz = T_5Hz * 5)
//   ggml layout [64, T_25Hz]: element (c, t) = data[t * 64 + c]
static int detok_ggml_decode(DetokGGML * m, const int * codes, int T_5Hz,
                              float * context_out) {
    int T_25Hz = T_5Hz * 5;
    int H = 2048;
    int P = 5;  // pool window: each 5Hz token -> 5 frames at 25Hz

    // Step 1: FSQ decode all indices on CPU -> [T_5Hz, 6]
    std::vector<float> fsq_decoded(T_5Hz * FSQ_NDIMS);
    for (int g = 0; g < T_5Hz; g++)
        fsq_decode_index(codes[g], fsq_decoded.data() + g * FSQ_NDIMS);

    // Step 2: build ggml graph for one token
    // input [6] -> project_out [2048] -> embed_tokens [2048]
    //   -> broadcast + special_tokens [2048, 5] -> 2L encoder -> norm -> proj_out [64, 5]
    // Graph context (generous fixed allocation)
    size_t ctx_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(4096, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    struct ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    struct ggml_context * ctx = ggml_init(p);

    // Input: one FSQ-decoded vector [6]
    // ggml pitfall: [6] is ne[0]=6, matches project_out weight [2048, 6]
    struct ggml_tensor * fsq_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FSQ_NDIMS);
    ggml_set_name(fsq_in, "fsq_in");
    ggml_set_input(fsq_in);

    // project_out: [6] -> [2048]
    struct ggml_tensor * quantized = ggml_mul_mat(ctx, m->fsq_proj_w, fsq_in);
    quantized = ggml_add(ctx, quantized, qwen3_f32(ctx, m->fsq_proj_b));

    // embed_tokens: [2048] -> [2048]
    struct ggml_tensor * embedded = ggml_mul_mat(ctx, m->embed_w,
                                                  ggml_reshape_2d(ctx, quantized, H, 1));
    embedded = ggml_add(ctx, embedded, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->embed_b, H, 1)));

    // Broadcast [2048, 1] -> [2048, 5] + special_tokens [2048, 5]
    // ggml pitfall: special_tokens loaded as BF16, cast to F32 for add
    struct ggml_tensor * special_2d = ggml_reshape_2d(ctx, m->special_tok, H, P);
    struct ggml_tensor * special_f32 = qwen3_f32(ctx, special_2d);
    struct ggml_tensor * hidden = ggml_add(ctx, ggml_repeat(ctx, embedded, special_f32),
                                            special_f32);

    // Position indices for RoPE: [0, 1, 2, 3, 4]
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_set_name(positions, "detok_pos");
    ggml_set_input(positions);

    // 2L encoder + norm (non-causal, no mask needed at S=5)
    hidden = qwen3_build_layers(ctx, m->cfg, m->layers, m->norm,
                                 hidden, positions, NULL, P);

    // proj_out: [2048, 5] -> [64, 5]
    struct ggml_tensor * output = ggml_mul_mat(ctx, m->proj_out_w, hidden);
    output = ggml_add(ctx, output, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->proj_out_b, 64, 1)));
    ggml_set_name(output, "detok_out");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, output);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[Context] FATAL: graph alloc failed\n");
        ggml_free(ctx);
        return -1;
    }

    // Set positions once (constant: 0,1,2,3,4)
    int pos_data[5] = {0, 1, 2, 3, 4};
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "detok_pos"),
                            pos_data, 0, P * sizeof(int));

    struct ggml_tensor * t_in = ggml_graph_get_tensor(gf, "fsq_in");
    struct ggml_tensor * t_out = ggml_graph_get_tensor(gf, "detok_out");

    // Step 3: loop over T_5Hz tokens
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "detok_pos");
    for (int g = 0; g < T_5Hz; g++) {
        // Re-set positions every iteration (allocator may share buffer with intermediates)
        ggml_backend_tensor_set(t_pos, pos_data, 0, P * sizeof(int));
        ggml_backend_tensor_set(t_in, fsq_decoded.data() + g * FSQ_NDIMS,
                                0, FSQ_NDIMS * sizeof(float));
        ggml_backend_sched_graph_compute(m->sched, gf);

        // output [64, 5]: 5 frames of 64 channels
        // context_out layout: [64, T_25Hz], frame t at offset t*64
        ggml_backend_tensor_get(t_out, context_out + g * P * 64,
                                0, P * 64 * sizeof(float));
    }

    fprintf(stderr, "[Context] Decoded: %d codes -> %d frames (%.1fs @ 25Hz)\n",
            T_5Hz, T_25Hz, (float)T_25Hz / 25.0f);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    return T_25Hz;
}

// Free
static void detok_ggml_free(DetokGGML * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    wctx_free(&m->wctx);
    *m = {};
}

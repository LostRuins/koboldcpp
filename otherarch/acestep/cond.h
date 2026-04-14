// cond.h: ACEStep Condition Encoder via ggml
//
// Produces encoder_hidden_states [S_total, 2048] from (all arrays H-contiguous per token):
//   - text_hidden [S_text, 1024]   : from Qwen3-Embedding text encoder
//   - lyric_embed [S_lyric, 1024]  : from CPU vocab lookup of lyric tokens
//   - timbre_feats [S_ref, 64]     : reference audio features (optional)
//
// Internal pipeline (ggml notation [ne0, ne1]):
//   text_hidden  -> Linear(1024->2048)                     -> [2048, S_text]
//   lyric_embed  -> Linear(1024->2048)+bias  -> 8L -> norm -> [2048, S_lyric]
//   timbre_feats -> Linear(64->2048)+bias    -> 4L -> norm -> take frame[0] -> [2048, 1]
//   Pack: cat(lyric, timbre[0:1], text_proj) -> [2048, S_total]

#pragma once
#include "qwen3.h"

// Lyric/Timbre encoder configs
static Qwen3Config qwen3_lyric_config() {
    return {
        /*hidden_size*/       2048,
        /*intermediate_size*/ 6144,
        /*n_heads*/           16,
        /*n_kv_heads*/        8,
        /*head_dim*/          128,
        /*n_layers*/          8,
        /*rope_theta*/        1000000.0f,
        /*rms_norm_eps*/      1e-6f,
        /*is_causal*/         false,
    };
}

static Qwen3Config qwen3_timbre_config() {
    return {
        /*hidden_size*/       2048,
        /*intermediate_size*/ 6144,
        /*n_heads*/           16,
        /*n_kv_heads*/        8,
        /*head_dim*/          128,
        /*n_layers*/          4,
        /*rope_theta*/        1000000.0f,
        /*rms_norm_eps*/      1e-6f,
        /*is_causal*/         false,
    };
}

// Struct
struct CondGGML {
    // Lyric encoder (8L, H=2048)
    Qwen3Config lyric_cfg;
    Qwen3Layer  lyric_layers[8];
    struct ggml_tensor * lyric_embed_w;   // [2048, 1024] ggml = Linear(1024->2048)
    struct ggml_tensor * lyric_embed_b;   // [2048]
    struct ggml_tensor * lyric_norm;      // [2048]

    // Timbre encoder (4L, H=2048)
    Qwen3Config timbre_cfg;
    Qwen3Layer  timbre_layers[4];
    struct ggml_tensor * timbre_embed_w;  // [2048, 64] ggml = Linear(64->2048)
    struct ggml_tensor * timbre_embed_b;  // [2048]
    struct ggml_tensor * timbre_norm;     // [2048]

    // Text projector: Linear(1024->2048) no bias
    struct ggml_tensor * text_proj_w;     // [2048, 1024] ggml

    // Null condition embedding (for classifier-free guidance)
    struct ggml_tensor * null_cond_emb;   // [2048, 1, 1]

    // Backend
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx wctx;
};

// Init
static void cond_ggml_init_backend(CondGGML * m) {
    BackendPair bp = backend_init("CondEncoder");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
}

// Load from ACEStep DiT GGUF
// gguf_path: path to the DiT .gguf file
// Tensors have prefix "encoder." for lyric/timbre, and "null_condition_emb"
static bool cond_ggml_load(CondGGML * m, const char * gguf_path) {
    m->lyric_cfg  = qwen3_lyric_config();
    m->timbre_cfg = qwen3_timbre_config();

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // Count tensors:
    // lyric: embed_w(1) + embed_b(1) + 8 layers x 11(88) + norm(1) = 91
    // timbre: embed_w(1) + embed_b(1) + 4 layers x 11(44) + norm(1) = 47
    // text_proj(1) + null_cond(1) = 2
    // Total: 140
    int n_tensors = 91 + 47 + 2;
    wctx_init(&m->wctx, n_tensors);

    // Lyric encoder
    m->lyric_embed_w = gf_load_tensor(&m->wctx, gf, "encoder.lyric_encoder.embed_tokens.weight");
    m->lyric_embed_b = gf_load_tensor_f32(&m->wctx, gf, "encoder.lyric_encoder.embed_tokens.bias");
    m->lyric_norm    = gf_load_tensor_f32(&m->wctx, gf, "encoder.lyric_encoder.norm.weight");
    fprintf(stderr, "[Load] LyricEncoder: %dL\n", m->lyric_cfg.n_layers);
    for (int i = 0; i < m->lyric_cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "encoder.lyric_encoder.layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->lyric_layers[i], prefix, i);
    }

    // Timbre encoder
    m->timbre_embed_w = gf_load_tensor(&m->wctx, gf, "encoder.timbre_encoder.embed_tokens.weight");
    m->timbre_embed_b = gf_load_tensor_f32(&m->wctx, gf, "encoder.timbre_encoder.embed_tokens.bias");
    m->timbre_norm    = gf_load_tensor_f32(&m->wctx, gf, "encoder.timbre_encoder.norm.weight");
    fprintf(stderr, "[Load] TimbreEncoder: %dL\n", m->timbre_cfg.n_layers);
    for (int i = 0; i < m->timbre_cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "encoder.timbre_encoder.layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->timbre_layers[i], prefix, i);
    }

    // Text projector + null condition
    m->text_proj_w   = gf_load_tensor(&m->wctx, gf, "encoder.text_projector.weight");
    m->null_cond_emb = gf_load_tensor(&m->wctx, gf, "null_condition_emb");

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[Load] CondEncoder: lyric(%dL), timbre(%dL), text_proj, null_cond\n",
            m->lyric_cfg.n_layers, m->timbre_cfg.n_layers);
    return true;
}

// Forward
//
// Inputs (CPU float arrays):
//   text_hidden:  [1024 * S_text]   from text encoder (Qwen3-Embedding)
//   lyric_embed:  [1024 * S_lyric]  from CPU vocab lookup of lyric tokens
//   timbre_feats: [64 * S_ref]      reference audio features (NULL if none)
//   S_text, S_lyric, S_ref          sequence lengths
//
// Output:
//   enc_hidden:   resized to [2048 * S_total] float
//   *out_enc_S:   total sequence length
//
// Layout: all arrays in ggml order (ne[0]=dim contiguous, then sequence)
static void cond_ggml_forward(CondGGML * m,
                               const float * text_hidden, int S_text,
                               const float * lyric_embed, int S_lyric,
                               const float * timbre_feats, int S_ref,
                               std::vector<float> & enc_hidden,
                               int * out_enc_S) {
    int H = 2048;
    bool has_timbre = (timbre_feats != NULL && S_ref > 0);

    // Graph context (generous fixed allocation)
    size_t ctx_size = 4096 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Positions for lyric (bidirectional, 0..S_lyric-1)
    struct ggml_tensor * lyric_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_lyric);
    ggml_set_name(lyric_pos, "lyric_pos");
    ggml_set_input(lyric_pos);

    // Path 1: Lyric encoder
    struct ggml_tensor * t_lyric_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, S_lyric);
    ggml_set_name(t_lyric_in, "lyric_in");
    ggml_set_input(t_lyric_in);

    // Linear embed: [1024, S_lyric] -> [2048, S_lyric]
    struct ggml_tensor * lyric_h = qwen3_linear_bias(ctx, m->lyric_embed_w,
                                                      m->lyric_embed_b, t_lyric_in);

    // Bidirectional sliding window mask for even layers (|i-j| <= 128)
    // Python: layer_types = [sliding, full, sliding, full, ...]
    // sliding_window = 128
    struct ggml_tensor * lyric_slide_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S_lyric, S_lyric);
    ggml_set_name(lyric_slide_mask, "lyric_slide_mask");
    ggml_set_input(lyric_slide_mask);

    // 8 layers with alternating masks + final norm
    for (int i = 0; i < m->lyric_cfg.n_layers; i++) {
        struct ggml_tensor * layer_mask = (i % 2 == 0) ? lyric_slide_mask : NULL;
        lyric_h = qwen3_build_layer(ctx, m->lyric_cfg, &m->lyric_layers[i],
                                     lyric_h, lyric_pos, layer_mask, S_lyric);
    }
    lyric_h = qwen3_rms_norm(ctx, lyric_h, m->lyric_norm, m->lyric_cfg.rms_norm_eps);

    ggml_set_name(lyric_h, "lyric_out");
    ggml_set_output(lyric_h);

    // Path 2: Text projection
    struct ggml_tensor * t_text_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, S_text);
    ggml_set_name(t_text_in, "text_in");
    ggml_set_input(t_text_in);

    // Linear: [1024, S_text] -> [2048, S_text]
    struct ggml_tensor * text_proj = qwen3_linear(ctx, m->text_proj_w, t_text_in);
    ggml_set_name(text_proj, "text_proj_out");
    ggml_set_output(text_proj);

    // Path 3: Timbre encoder (optional)
    struct ggml_tensor * timbre_out = NULL;
    struct ggml_tensor * t_timbre_in = NULL;
    struct ggml_tensor * timbre_pos = NULL;
    struct ggml_tensor * timbre_slide_mask = NULL;

    if (has_timbre) {
        timbre_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_ref);
        ggml_set_name(timbre_pos, "timbre_pos");
        ggml_set_input(timbre_pos);

        t_timbre_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, S_ref);
        ggml_set_name(t_timbre_in, "timbre_in");
        ggml_set_input(t_timbre_in);

        // Linear embed: [64, S_ref] -> [2048, S_ref]
        struct ggml_tensor * timbre_h = qwen3_linear_bias(ctx, m->timbre_embed_w,
                                                           m->timbre_embed_b, t_timbre_in);

        // Bidirectional sliding window mask for even layers
        timbre_slide_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S_ref, S_ref);
        ggml_set_name(timbre_slide_mask, "timbre_slide_mask");
        ggml_set_input(timbre_slide_mask);

        // 4 layers with alternating masks + final norm
        for (int i = 0; i < m->timbre_cfg.n_layers; i++) {
            struct ggml_tensor * layer_mask = (i % 2 == 0) ? timbre_slide_mask : NULL;
            timbre_h = qwen3_build_layer(ctx, m->timbre_cfg, &m->timbre_layers[i],
                                          timbre_h, timbre_pos, layer_mask, S_ref);
        }
        timbre_h = qwen3_rms_norm(ctx, timbre_h, m->timbre_norm, m->timbre_cfg.rms_norm_eps);

        // Take first frame: [2048, S_ref] -> view [2048, 1]
        timbre_out = ggml_view_2d(ctx, timbre_h, H, 1,
                                   timbre_h->nb[1], 0);
        ggml_set_name(timbre_out, "timbre_out");
        ggml_set_output(timbre_out);
    }

    // Build forward
    ggml_build_forward_expand(gf, lyric_h);
    ggml_build_forward_expand(gf, text_proj);
    if (timbre_out) ggml_build_forward_expand(gf, timbre_out);

    // Allocate and set inputs
    ggml_backend_sched_alloc_graph(m->sched, gf);

    ggml_backend_tensor_set(t_lyric_in, lyric_embed, 0, 1024 * S_lyric * sizeof(float));
    ggml_backend_tensor_set(t_text_in, text_hidden, 0, 1024 * S_text * sizeof(float));

    // Lyric positions
    {
        std::vector<int> pos(S_lyric);
        for (int i = 0; i < S_lyric; i++) pos[i] = i;
        ggml_backend_tensor_set(lyric_pos, pos.data(), 0, S_lyric * sizeof(int));
    }

    // Lyric sliding window mask: bidirectional, |i-j| <= 128
    {
        const int W = 128;
        std::vector<uint16_t> mask_data(S_lyric * S_lyric);
        for (int i = 0; i < S_lyric; i++) {
            for (int j = 0; j < S_lyric; j++) {
                int d = i - j; if (d < 0) d = -d;
                mask_data[i * S_lyric + j] = ggml_fp32_to_fp16(d <= W ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(lyric_slide_mask, mask_data.data(), 0,
                                 S_lyric * S_lyric * sizeof(uint16_t));
        fprintf(stderr, "[CondEnc] Lyric sliding mask: %dx%d, window=%d\n", S_lyric, S_lyric, W);
    }

    if (has_timbre) {
        ggml_backend_tensor_set(t_timbre_in, timbre_feats, 0, 64 * S_ref * sizeof(float));
        std::vector<int> pos(S_ref);
        for (int i = 0; i < S_ref; i++) pos[i] = i;
        ggml_backend_tensor_set(timbre_pos, pos.data(), 0, S_ref * sizeof(int));

        // Timbre sliding window mask: bidirectional, |i-j| <= 128
        const int W = 128;
        std::vector<uint16_t> mask_data(S_ref * S_ref);
        for (int i = 0; i < S_ref; i++) {
            for (int j = 0; j < S_ref; j++) {
                int d = i - j; if (d < 0) d = -d;
                mask_data[i * S_ref + j] = ggml_fp32_to_fp16(d <= W ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(timbre_slide_mask, mask_data.data(), 0,
                                 S_ref * S_ref * sizeof(uint16_t));
        fprintf(stderr, "[CondEnc] Timbre sliding mask: %dx%d, window=%d\n", S_ref, S_ref, W);
    }

    // Compute
    ggml_backend_sched_graph_compute(m->sched, gf);

    // Read outputs and pack on CPU
    // Pack order: lyric, timbre[0:1], text_proj
    int S_timbre_out = has_timbre ? 1 : 0;
    int S_total = S_lyric + S_timbre_out + S_text;
    enc_hidden.resize(H * S_total);
    *out_enc_S = S_total;

    int offset = 0;

    // Lyric: [2048, S_lyric]
    ggml_backend_tensor_get(lyric_h, enc_hidden.data() + offset * H,
                            0, H * S_lyric * sizeof(float));
    offset += S_lyric;

    // Timbre: [2048, 1]
    if (timbre_out) {
        ggml_backend_tensor_get(timbre_out, enc_hidden.data() + offset * H,
                                0, H * 1 * sizeof(float));
        offset += 1;
    }

    // Text projection: [2048, S_text]
    ggml_backend_tensor_get(text_proj, enc_hidden.data() + offset * H,
                            0, H * S_text * sizeof(float));
    offset += S_text;

    fprintf(stderr, "[Encode] Packed: lyric=%d + timbre=%d + text=%d = %d tokens\n",
            S_lyric, S_timbre_out, S_text, S_total);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

// Free
static void cond_ggml_free(CondGGML * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
// vae.h: AutoencoderOobleck decoder (audio VAE) via ggml
//
// Architecture: conv1(64->2048,k=7) -> 5xblock(snake+convT+3xresunit) -> snake+conv2(128->2,k=7)
// ResUnit(ch, dil): skip=x -> snake->conv(k=7,dil)->snake->conv(k=1)->+skip
// Snake: x + sin^2(e^a * x) * (1/e^b)
// Weight norm fused at load: w = g*v/||v||
// Upsample: 10x6x4x4x2 = 1920x

#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_weights.h"
#include "backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Structs
struct VAEResUnit {
    struct ggml_tensor * s1a, * s1b;   // snake1 exp(alpha), exp(beta) [1, C]
    struct ggml_tensor * c1w, * c1b;   // conv1 fused [7, C, C], bias [C]
    struct ggml_tensor * s2a, * s2b;   // snake2
    struct ggml_tensor * c2w, * c2b;   // conv2 fused [1, C, C], bias [C]
    int dilation;
};

struct VAEBlock {
    struct ggml_tensor * sa, * sb;     // snake exp(a/b) [1, in_ch]
    struct ggml_tensor * ctw, * ctb;   // conv_transpose fused [K, out_ch, in_ch], bias [out_ch]
    int in_ch, out_ch, stride, kernel;
    VAEResUnit ru[3];
};

struct VAEGGML {
    struct ggml_tensor * c1w, * c1b;   // conv1 [7, 64, 2048], bias [2048]
    VAEBlock blk[5];
    struct ggml_tensor * sa, * sb;     // final snake [1, 128]
    struct ggml_tensor * c2w;          // conv2 [7, 128, 2] (no bias)

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    ggml_backend_buffer_t buf;
    struct ggml_context * weight_ctx;  // holds weight tensor metadata

    // Graph cache for tiled decode (avoids rebuild per tile)
    struct ggml_context * graph_ctx;
    uint8_t            * graph_buf;    // heap-allocated backing for graph_ctx
    struct ggml_cgraph  * graph;
    struct ggml_tensor  * graph_input;
    struct ggml_tensor  * graph_output;
    int                   graph_T;     // cached T_latent (0 = no cache)

    // Scratch buffer (reused across tiles, grown as needed)
    std::vector<float> scratch_in;     // transposed input [64 * T]
};

// Load helpers
// Fuse weight_norm: w = g*v/||v||, write f32 into pre-allocated ggml_tensor
// Works for both Conv1d [OC,IC,K] and ConvTranspose1d [IC,OC,K]:
// weight_norm normalizes over dim=0 (shape[0]), regardless of semantics.
static void vae_fuse_wn(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & pfx) {
    struct ggml_tensor * mv = ggml_get_tensor(gf.meta, (pfx + ".weight_v").c_str());
    const uint16_t * g = (const uint16_t *)gf_get_data(gf, (pfx + ".weight_g").c_str());
    const uint16_t * v = (const uint16_t *)gf_get_data(gf, (pfx + ".weight_v").c_str());
    // PyTorch dim0 is ggml ne[n_dims-1]
    int n_dims = ggml_n_dims(mv);
    int dim0 = (int)mv->ne[n_dims - 1];
    int fan = (int)(ggml_nelements(mv) / dim0);
    std::vector<float> w(dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&g[d]);
        float nsq = 0;
        for (int i = 0; i < fan; i++) {
            float vv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&v[d * fan + i]);
            nsq += vv * vv;
        }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) {
            float vv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&v[d * fan + i]);
            w[d * fan + i] = vv * s;
        }
    }
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> w16(w.size());
        ggml_fp32_to_fp16_row(w.data(), w16.data(), (int)w.size());
        ggml_backend_tensor_set(dst, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    }
}

// Load bf16 snake param [1,C,1] -> exp -> f32 [1, C]
static void vae_load_snake(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[1];  // PyTorch [1,C,1] -> ggml ne=[1,C,1], middle dim
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = expf(ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]));
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load bf16 snake param [1,C,1] -> 1/exp -> f32 [1, C] (reciprocal for mul fusion)
static void vae_load_snake_inv(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[1];
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = 1.0f / expf(ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]));
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load bf16 bias [C] -> f32
static void vae_load_bias(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[0];  // 1D: ne[0] = C
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]);
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load model
static void vae_ggml_load(VAEGGML * m, const char * path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        fprintf(stderr, "[VAE] FATAL: cannot load %s\n", path);
        exit(1);
    }

    static const int strides[]   = {10, 6, 4, 4, 2};
    static const int in_ch[]     = {2048, 1024, 512, 256, 128};
    static const int out_ch[]    = {1024, 512, 256, 128, 128};
    static const int dilations[] = {1, 3, 9};

    // Phase 1: create tensor metadata (no_alloc context)
    size_t ctx_size = ggml_tensor_overhead() * 200;
    struct ggml_init_params p = { ctx_size, NULL, true };
    m->weight_ctx = ggml_init(p);
    struct ggml_context * ctx = m->weight_ctx;

    m->c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 64, 2048);
    m->c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);

    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        b.in_ch  = in_ch[i];
        b.out_ch = out_ch[i];
        b.stride = strides[i];
        b.kernel = strides[i] * 2;
        int C    = out_ch[i];
        b.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.ctw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, b.kernel, out_ch[i], in_ch[i]);
        b.ctb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            ru.dilation = dilations[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    m->sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m->sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m->c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 128, 2);

    // Phase 2: allocate backend buffer (im2col grid Y fix enables long-sequence conv1d)
    BackendPair bp = backend_init("VAE");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
    m->buf = ggml_backend_alloc_ctx_tensors(ctx, m->backend);
    fprintf(stderr, "[VAE] Backend: %s, Weight buffer: %.1f MB\n",
            ggml_backend_name(m->backend),
            (float)ggml_backend_buffer_get_size(m->buf) / (1024 * 1024));

    // Phase 3: load & fuse weights
    vae_fuse_wn(m->c1w, gf, "decoder.conv1");
    vae_load_bias(m->c1b, gf, "decoder.conv1.bias");

    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        std::string blk_pfx = "decoder.block." + std::to_string(i);
        vae_load_snake(b.sa, gf, blk_pfx + ".snake1.alpha");
        vae_load_snake_inv(b.sb, gf, blk_pfx + ".snake1.beta");
        vae_fuse_wn(b.ctw, gf, blk_pfx + ".conv_t1");
        vae_load_bias(b.ctb, gf, blk_pfx + ".conv_t1.bias");
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            std::string rp = blk_pfx + ".res_unit" + std::to_string(r + 1);
            vae_load_snake(ru.s1a, gf, rp + ".snake1.alpha");
            vae_load_snake_inv(ru.s1b, gf, rp + ".snake1.beta");
            vae_fuse_wn(ru.c1w, gf, rp + ".conv1");
            vae_load_bias(ru.c1b, gf, rp + ".conv1.bias");
            vae_load_snake(ru.s2a, gf, rp + ".snake2.alpha");
            vae_load_snake_inv(ru.s2b, gf, rp + ".snake2.beta");
            vae_fuse_wn(ru.c2w, gf, rp + ".conv2");
            vae_load_bias(ru.c2b, gf, rp + ".conv2.bias");
        }
    }
    vae_load_snake(m->sa, gf, "decoder.snake1.alpha");
    vae_load_snake_inv(m->sb, gf, "decoder.snake1.beta");
    vae_fuse_wn(m->c2w, gf, "decoder.conv2");

    fprintf(stderr, "[VAE] Loaded: 5 blocks, upsample=1920x\n");
    gf_close(&gf);
}

// Graph building
// Snake activation: x + sin^2(exp_a * x) * inv_b
// x: [T, C], exp_a: [1, C], inv_b: [1, C] (pre-computed 1/exp(b))
static struct ggml_tensor * vae_snake(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        struct ggml_tensor * exp_a,
        struct ggml_tensor * inv_b) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, exp_a);      // [T, C] (broadcast 1->T)
    struct ggml_tensor * s  = ggml_sin(ctx, ax);             // sin(e^a * x)
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);              // sin^2
    struct ggml_tensor * d  = ggml_mul(ctx, s2, inv_b);      // * 1/e^b
    return ggml_add(ctx, x, d);                              // x + ...
}

// Conv1d + bias: data [T, IC] -> [T_out, OC]
static struct ggml_tensor * vae_conv1d(
        struct ggml_context * ctx,
        struct ggml_tensor * w,    // [K, IC, OC] (F16, pre-cast at load)
        struct ggml_tensor * b,    // [OC] or NULL
        struct ggml_tensor * x,    // [T, IC]
        int stride, int padding, int dilation) {
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    // ggml_conv_1d returns [OL, OC, N=1], squeeze to 2d
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        // bias [OC] -> [1, OC] for broadcast over OL dimension
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose1d + bias
static struct ggml_tensor * vae_conv_t1d(
        struct ggml_context * ctx,
        struct ggml_tensor * w,    // [K, out_ch, in_ch]
        struct ggml_tensor * b,    // [out_ch] or NULL
        struct ggml_tensor * x,    // [T, in_ch]
        int stride, int padding) {
    // ggml_conv_transpose_1d asserts p0==0, so we crop manually
    struct ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    // y is 4d: [OL, OC, 1, 1], squeeze to 2d
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    // Crop padding from both sides: [OL, OC] -> [OL - 2*pad, OC]
    if (padding > 0) {
        y = ggml_view_2d(ctx, y, y->ne[0] - 2 * padding, y->ne[1],
                         y->nb[1], padding * sizeof(float));
    }
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ResUnit forward
static struct ggml_tensor * vae_res_unit(
        struct ggml_context * ctx,
        VAEResUnit * ru,
        struct ggml_tensor * x) {   // [T, C]
    struct ggml_tensor * skip = x;

    // snake1 -> dilated conv(k=7) -> snake2 -> conv(k=1)
    int pad = 3 * ru->dilation;  // (k-1)*dil/2 = 3*dil
    x = vae_snake(ctx, x, ru->s1a, ru->s1b);
    x = vae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x = vae_snake(ctx, x, ru->s2a, ru->s2b);
    x = vae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);

    return ggml_add(ctx, skip, x);
}

// Build full VAE decode graph
// latent: [T_latent, 64] -> audio: [T_audio, 2]
static struct ggml_tensor * vae_ggml_build_graph(
        struct ggml_context * ctx,
        VAEGGML * m,
        struct ggml_tensor * latent) {   // [T, 64] input

    // conv1: [T, 64] -> [T, 2048]
    struct ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, latent, 1, 3, 1);

    // 5 decoder blocks
    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        // snake -> conv_transpose (upsample)
        x = vae_snake(ctx, x, b.sa, b.sb);
        int pad = (b.kernel - b.stride) / 2;
        x = vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad);
        // 3 res units
        for (int r = 0; r < 3; r++)
            x = vae_res_unit(ctx, &b.ru[r], x);
    }

    // Final: snake -> conv2(128->2, k=7, pad=3)
    x = vae_snake(ctx, x, m->sa, m->sb);
    x = vae_conv1d(ctx, m->c2w, NULL, x, 1, 3, 1);

    return x;  // [T_audio, 2]
}

// Core compute: ensure graph cached, set input, run. Returns T_audio or -1.
// Output remains in m->graph_output for caller to read as needed.
static int vae_ggml_compute(
        VAEGGML * m,
        const float * latent,   // [T_full, 64] time-major
        int T_latent,           // window length to decode
        int win_start = 0) {    // offset into latent

    // Build graph only when T_latent changes (cached for tiled decode reuse)
    if (m->graph_T != T_latent) {
        if (m->graph_ctx) {
            ggml_backend_sched_reset(m->sched);
            ggml_free(m->graph_ctx);
            free(m->graph_buf);
        }
        // Graph context (generous fixed allocation)
        size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
        m->graph_buf = (uint8_t *)malloc(ctx_size);
        struct ggml_init_params p = { ctx_size, m->graph_buf, true };
        struct ggml_context * ctx = ggml_init(p);

        m->graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, 64);
        ggml_set_name(m->graph_input, "vae_input");
        ggml_set_input(m->graph_input);

        m->graph_output = vae_ggml_build_graph(ctx, m, m->graph_input);
        ggml_set_name(m->graph_output, "vae_output");
        ggml_set_output(m->graph_output);

        m->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(m->graph, m->graph_output);

        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {
            fprintf(stderr, "[VAE] FATAL: graph alloc failed\n");
            ggml_free(ctx);
            free(m->graph_buf);
            m->graph_ctx = NULL;
            m->graph_buf = NULL;
            m->graph_T = 0;
            return -1;
        }
        m->graph_ctx = ctx;
        m->graph_T = T_latent;
        fprintf(stderr, "[VAE] Graph: %d nodes, T_latent=%d\n",
                ggml_graph_n_nodes(m->graph), T_latent);
    }

    // Extract window + transpose: [T, 64] time-major -> ggml [T, 64] channel-major
    size_t in_size = 64 * T_latent;
    if (m->scratch_in.size() < in_size)
        m->scratch_in.resize(in_size);
    for (int c = 0; c < 64; c++)
        for (int t = 0; t < T_latent; t++)
            m->scratch_in[c * T_latent + t] = latent[(win_start + t) * 64 + c];
    ggml_backend_tensor_set(m->graph_input,
                            m->scratch_in.data(), 0, in_size * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, m->graph);

    return (int)m->graph_output->ne[0];
}

// Decode API: latent [T_latent, 64] -> audio [2, T_audio] flat.
// Returns T_audio (or -1 on error).
static int vae_ggml_decode(
        VAEGGML * m,
        const float * latent,
        int T_latent,
        float * audio_out,
        int max_T_audio) {

    int T_audio = T_latent * 1920;
    if (T_audio > max_T_audio) {
        fprintf(stderr, "[VAE] T_audio %d exceeds max %d\n", T_audio, max_T_audio);
        return -1;
    }

    int T_out = vae_ggml_compute(m, latent, T_latent, 0);
    if (T_out < 0) return -1;

    ggml_backend_tensor_get(m->graph_output, audio_out, 0, T_out * 2 * sizeof(float));

    fprintf(stderr, "[VAE] Decoded: T_latent=%d -> T_audio=%d (%.2fs @ 48kHz)\n",
            T_latent, T_out, (float)T_out / 48000.0f);
    return T_out;
}

// Tiled decode: overlap-discard chunking for bounded VRAM usage.
// Matches Python handler.py tiled_decode / _tiled_decode_gpu:
//   stride = chunk_size - 2*overlap
//   For each tile: decode latent window with overlap context, trim to core, concatenate.
// Default chunk=256, overlap=64 matches Python handler.py fallback defaults.
// Python auto-tunes chunk by VRAM: >=24GB->512, >=16GB->384, >=12GB->256, <12GB->128.
// Returns T_audio (total samples per channel) or -1 on error.
static int vae_ggml_decode_tiled(
        VAEGGML * m,
        const float * latent,       // [T_latent, 64] flat time-major (DiT output layout)
        int T_latent,
        float * audio_out,          // [2, T_audio] flat (caller allocs)
        int max_T_audio,
        int chunk_size = 256,
        int overlap = 64) {

    // Ensure positive stride (matches Python effective_overlap reduction)
    while (chunk_size - 2 * overlap <= 0 && overlap > 0)
        overlap /= 2;

    // Short sequence: decode directly
    if (T_latent <= chunk_size)
        return vae_ggml_decode(m, latent, T_latent, audio_out, max_T_audio);

    int stride = chunk_size - 2 * overlap;
    int num_steps = (T_latent + stride - 1) / stride;

    fprintf(stderr, "[VAE] Tiled decode: %d tiles (chunk=%d, overlap=%d, stride=%d)\n",
            num_steps, chunk_size, overlap, stride);

    // FIX (Bug 2): Zero the ch1 staging zone before the loop so stale data from a
    // previous generation (e.g. after a lowvram reload, or a shorter re-run into the
    // same buffer) cannot bleed into gaps between tile writes and corrupt the memmove.
    memset(audio_out + max_T_audio, 0, max_T_audio * sizeof(float));

    float upsample_factor = 0.0f;
    int audio_write_pos = 0;

    for (int i = 0; i < num_steps; i++) {
        // Core range in latent frames (the part we keep)
        int core_start = i * stride;
        int core_end = core_start + stride;
        if (core_end > T_latent) core_end = T_latent;

        // Window range with overlap context
        int win_start = core_start - overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + overlap;
        if (win_end > T_latent) win_end = T_latent;
        int win_len = win_end - win_start;

        // Compute tile (graph cached, extract+transpose fused)
        int tile_T = vae_ggml_compute(m, latent, win_len, win_start);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE] FATAL: tile %d decode failed\n", i);
            return -1;
        }

        // Determine upsample factor from first tile
        if (i == 0) {
            upsample_factor = (float)tile_T / (float)win_len;
            fprintf(stderr, "[VAE] Upsample factor: %.2f (expected ~1920)\n", upsample_factor);
        }

        // Compute trim in audio samples (matches Python int(round(...)))
        int added_start = core_start - win_start;
        int trim_start = (int)roundf((float)added_start * upsample_factor);
        int added_end = win_end - core_end;
        int trim_end = (int)roundf((float)added_end * upsample_factor);

        int end_idx = (trim_end > 0) ? (tile_T - trim_end) : tile_T;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        // Check output bounds
        if (audio_write_pos + core_len > max_T_audio) {
            fprintf(stderr, "[VAE] FATAL: tiled output exceeds max_T_audio\n");
            return -1;
        }

        // Read trimmed ch0 and ch1 directly from backend tensor into final audio_out
        // Layout: [ch0: tile_T floats, ch1: tile_T floats]
        // FIX (Bug 1): Capture tile_T into a local before the two reads so both
        // ch0 and ch1 use the same tile's stride. (Guards against any future refactor
        // that might clobber tile_T between the two ggml_backend_tensor_get calls.)
        int this_tile_T = tile_T;
        ggml_backend_tensor_get(m->graph_output,
                                audio_out + audio_write_pos,
                                trim_start * sizeof(float),
                                core_len * sizeof(float));
        ggml_backend_tensor_get(m->graph_output,
                                audio_out + max_T_audio + audio_write_pos,
                                (this_tile_T + trim_start) * sizeof(float),
                                core_len * sizeof(float));
        audio_write_pos += core_len;
    }

    // Compact ch1 from offset max_T_audio to offset audio_write_pos
    memmove(audio_out + audio_write_pos,
            audio_out + max_T_audio,
            audio_write_pos * sizeof(float));

    fprintf(stderr, "[VAE] Tiled decode done: %d tiles -> T_audio=%d (%.2fs @ 48kHz)\n",
            num_steps, audio_write_pos, (float)audio_write_pos / 48000.0f);

    return audio_write_pos;
}

// Free
static void vae_ggml_free(VAEGGML * m) {
    if (m->graph_ctx) {
        ggml_backend_sched_reset(m->sched);
        ggml_free(m->graph_ctx);
        free(m->graph_buf);
    }
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->buf) ggml_backend_buffer_free(m->buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    *m = {};
}

// vae-enc.h: AutoencoderOobleck encoder (audio -> latent) via ggml
//
// Mirror of vae.h decoder. Reuses VAEResUnit, load helpers, graph ops.
// Architecture: conv1(2->128,k=7) -> 5x(3xresunit+snake+strided_conv) -> snake+conv2(2048->128,k=3)
// Output 128ch = mean[64] + scale[64]. Deterministic encode returns mean.
// Downsample: 2x4x4x6x10 = 1920x (matches decoder upsample)

#pragma once
#include "vae.h"

// Encoder block: 3xResUnit(in_ch) -> snake(in_ch) -> strided Conv1d(in_ch -> out_ch)
// Decoder block is the mirror: snake(in_ch) -> ConvT(in_ch -> out_ch) -> 3xResUnit(out_ch)
struct VAEEncBlock {
    VAEResUnit ru[3];
    struct ggml_tensor * sa, * sb;       // snake [1, in_ch]
    struct ggml_tensor * dw, * db;       // strided conv [K, in_ch, out_ch], bias [out_ch]
    int in_ch, out_ch, stride, kernel, padding;
};

struct VAEEncoder {
    struct ggml_tensor * c1w, * c1b;     // conv1 [7, 2, 128], bias [128]
    VAEEncBlock blk[5];
    struct ggml_tensor * sa, * sb;       // final snake [1, 2048]
    struct ggml_tensor * c2w, * c2b;     // conv2 [3, 2048, 128], bias [128]

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    ggml_backend_buffer_t buf;
    struct ggml_context * weight_ctx;

    // graph cache (rebuilt when T_audio changes)
    struct ggml_context * graph_ctx;
    uint8_t            * graph_buf;
    struct ggml_cgraph  * graph;
    struct ggml_tensor  * graph_input;   // [T_audio, 2]
    struct ggml_tensor  * graph_output;  // [T_latent, 128]
    int                   graph_T;       // cached T_audio (0 = no cache)

    std::vector<float> scratch_in;       // transposed input [2 * T_audio]
};

// Load encoder weights from the same VAE GGUF (encoder.* tensors)
static void vae_enc_load(VAEEncoder * m, const char * path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        fprintf(stderr, "[VAE-Enc] FATAL: cannot load %s\n", path);
        exit(1);
    }

    // Encoder channel layout (mirror of decoder, bottom-up):
    //   conv1: 2 -> 128
    //   block: [128->128, 128->256, 256->512, 512->1024, 1024->2048]
    //   conv2: 2048 -> 128 (split: mean[64] + scale[64])
    // ResUnits run at in_ch (before downsample), unlike decoder (at out_ch, after upsample).
    static const int in_ch[]   = {128, 128, 256,  512, 1024};
    static const int out_ch[]  = {128, 256, 512, 1024, 2048};
    static const int strides[] = {  2,   4,   4,    6,   10};
    static const int dilations[] = {1, 3, 9};

    // Phase 1: create weight tensors
    size_t ctx_size = ggml_tensor_overhead() * 256;
    struct ggml_init_params p = { ctx_size, NULL, true };
    m->weight_ctx = ggml_init(p);
    struct ggml_context * ctx = m->weight_ctx;

    m->c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 2, 128);
    m->c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    for (int i = 0; i < 5; i++) {
        VAEEncBlock & b = m->blk[i];
        b.in_ch   = in_ch[i];
        b.out_ch  = out_ch[i];
        b.stride  = strides[i];
        b.kernel  = strides[i] * 2;
        b.padding = (strides[i] + 1) / 2;   // ceil(stride / 2)
        int C = in_ch[i];                    // res_units + snake at in_ch

        // 3 res units at in_ch
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

        // snake at in_ch (before downsample conv)
        b.sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);

        // strided conv1d: [K, in_ch, out_ch]
        b.dw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kernel, in_ch[i], out_ch[i]);
        b.db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
    }

    m->sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m->sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m->c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 2048, 128);
    m->c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    // Phase 2: allocate backend buffer
    BackendPair bp = backend_init("VAE-Enc");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
    m->buf = ggml_backend_alloc_ctx_tensors(ctx, m->backend);
    if (!m->buf) {
        fprintf(stderr, "[VAE-Enc] FATAL: failed to allocate weight buffer\n");
        exit(1);
    }
    fprintf(stderr, "[VAE-Enc] Backend: %s, Weight buffer: %.1f MB\n",
            ggml_backend_name(m->backend),
            (float)ggml_backend_buffer_get_size(m->buf) / (1024 * 1024));

    // Phase 3: load and fuse weights
    vae_fuse_wn(m->c1w, gf, "encoder.conv1");
    vae_load_bias(m->c1b, gf, "encoder.conv1.bias");

    for (int i = 0; i < 5; i++) {
        VAEEncBlock & b = m->blk[i];
        std::string blk_pfx = "encoder.block." + std::to_string(i);

        // res_units first (same load pattern as decoder)
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

        // snake + strided downsample conv (regular conv1d, NOT transposed)
        vae_load_snake(b.sa, gf, blk_pfx + ".snake1.alpha");
        vae_load_snake_inv(b.sb, gf, blk_pfx + ".snake1.beta");
        vae_fuse_wn(b.dw, gf, blk_pfx + ".conv1");
        vae_load_bias(b.db, gf, blk_pfx + ".conv1.bias");
    }

    vae_load_snake(m->sa, gf, "encoder.snake1.alpha");
    vae_load_snake_inv(m->sb, gf, "encoder.snake1.beta");
    vae_fuse_wn(m->c2w, gf, "encoder.conv2");
    vae_load_bias(m->c2b, gf, "encoder.conv2.bias");

    fprintf(stderr, "[VAE-Enc] Loaded: 5 blocks, downsample=1920x, F32 activations\n");
    gf_close(&gf);
}

// Build encoder graph: audio [T_audio, 2] -> [T_latent, 128]
static struct ggml_tensor * vae_enc_build_graph(
        struct ggml_context * ctx,
        VAEEncoder * m,
        struct ggml_tensor * audio) {   // [T, 2]

    // conv1: [T, 2] -> [T, 128]
    struct ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, audio, 1, 3, 1);

    // 5 encoder blocks: resunits(in_ch) -> snake(in_ch) -> strided conv(in_ch -> out_ch)
    for (int i = 0; i < 5; i++) {
        VAEEncBlock & b = m->blk[i];
        for (int r = 0; r < 3; r++)
            x = vae_res_unit(ctx, &b.ru[r], x);
        x = vae_snake(ctx, x, b.sa, b.sb);
        x = vae_conv1d(ctx, b.dw, b.db, x, b.stride, b.padding, 1);
    }

    // Final: snake(2048) -> conv2(2048 -> 128, k=3, pad=1)
    x = vae_snake(ctx, x, m->sa, m->sb);
    x = vae_conv1d(ctx, m->c2w, m->c2b, x, 1, 1, 1);

    return x;  // [T_latent, 128]
}

// Core compute: build/cache graph, set input, run. Returns T_latent or -1.
// Output stays in m->graph_output for caller to read.
static int vae_enc_compute(
        VAEEncoder * m,
        const float * audio,    // [T_audio, 2] time-major interleaved stereo
        int T_audio) {

    // Rebuild graph when T_audio changes
    if (m->graph_T != T_audio) {
        if (m->graph_ctx) {
            ggml_backend_sched_reset(m->sched);
            ggml_free(m->graph_ctx);
            free(m->graph_buf);
        }

        size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
        m->graph_buf = (uint8_t *)malloc(ctx_size);
        struct ggml_init_params p = { ctx_size, m->graph_buf, true };
        struct ggml_context * ctx = ggml_init(p);

        m->graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_audio, 2);
        ggml_set_name(m->graph_input, "enc_input");
        ggml_set_input(m->graph_input);

        m->graph_output = vae_enc_build_graph(ctx, m, m->graph_input);
        ggml_set_name(m->graph_output, "enc_output");
        ggml_set_output(m->graph_output);

        m->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(m->graph, m->graph_output);

        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {
            fprintf(stderr, "[VAE-Enc] FATAL: graph alloc failed for T=%d\n", T_audio);
            ggml_free(ctx);
            free(m->graph_buf);
            m->graph_ctx = NULL;
            m->graph_buf = NULL;
            m->graph_T = 0;
            return -1;
        }

        m->graph_ctx = ctx;
        m->graph_T = T_audio;
        fprintf(stderr, "[VAE-Enc] Graph: %d nodes, T_audio=%d\n",
                ggml_graph_n_nodes(m->graph), T_audio);
    }

    // Transpose: [T, 2] time-major -> ggml [T, 2] channel-contiguous
    // ggml ne[0]=T is the contiguous dim, so we write all T samples per channel
    size_t in_size = (size_t)2 * T_audio;
    if (m->scratch_in.size() < in_size)
        m->scratch_in.resize(in_size);
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < T_audio; t++)
            m->scratch_in[c * T_audio + t] = audio[t * 2 + c];
    ggml_backend_tensor_set(m->graph_input,
                            m->scratch_in.data(), 0, in_size * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, m->graph);

    return (int)m->graph_output->ne[0];  // T_latent
}

// Encode API: audio [T_audio, 2] -> latent_out [T_latent, 64] (mean only, deterministic)
// Returns T_latent (or -1 on error).
// latent_out must hold at least (T_audio / 1920) * 64 floats.
static int vae_enc_encode(
        VAEEncoder * m,
        const float * audio,     // [T_audio, 2] interleaved stereo
        int T_audio,
        float * latent_out,      // [T_latent, 64] output, time-major
        int max_T_latent) {

    int T_latent = vae_enc_compute(m, audio, T_audio);
    if (T_latent < 0) return -1;

    if (T_latent > max_T_latent) {
        fprintf(stderr, "[VAE-Enc] T_latent %d exceeds max %d\n", T_latent, max_T_latent);
        return -1;
    }

    // Graph output is [ne0=T_latent, ne1=128] in ggml, channel-contiguous.
    // Channels 0..63 = mean, 64..127 = scale. We only read mean.
    // ggml layout: data[c * T_latent + t] for channel c, time t.
    // We write time-major: latent_out[t * 64 + c] = data[c * T_latent + t]
    //
    // Read the full 128ch output once, extract mean channels 0..63
    size_t out_bytes = (size_t)128 * T_latent * sizeof(float);
    std::vector<float> raw(128 * T_latent);
    ggml_backend_tensor_get(m->graph_output, raw.data(), 0, out_bytes);

    for (int t = 0; t < T_latent; t++)
        for (int c = 0; c < 64; c++)
            latent_out[t * 64 + c] = raw[c * T_latent + t];

    fprintf(stderr, "[VAE-Enc] Encode: T_audio=%d -> T_latent=%d (%.2fs @ 48kHz)\n",
            T_audio, T_latent, (float)T_audio / 48000.0f);

    return T_latent;
}

// Tiled encode for long audio (same chunking strategy as decoder)
// chunk_size: latent frames per tile, overlap: context frames on each side
static int vae_enc_encode_tiled(
        VAEEncoder * m,
        const float * audio,     // [T_audio, 2] interleaved stereo
        int T_audio,
        float * latent_out,      // [T_latent, 64] output, time-major
        int max_T_latent,
        int chunk_size = 256,
        int overlap = 64) {

    // Work in audio-sample space. Each latent frame = 1920 audio samples.
    int audio_chunk = chunk_size * 1920;
    int audio_overlap = overlap * 1920;

    // Shrink overlap until stride is positive
    while (audio_chunk - 2 * audio_overlap <= 0 && audio_overlap > 0)
        audio_overlap /= 2;

    // Short audio: encode directly
    if (T_audio <= audio_chunk)
        return vae_enc_encode(m, audio, T_audio, latent_out, max_T_latent);

    int audio_stride = audio_chunk - 2 * audio_overlap;
    int num_steps = (T_audio + audio_stride - 1) / audio_stride;

    fprintf(stderr, "[VAE-Enc] Tiled encode: %d tiles (chunk=%d, overlap=%d, stride=%d audio samples)\n",
            num_steps, audio_chunk, audio_overlap, audio_stride);

    float downsample_factor = 0.0f;
    int latent_write_pos = 0;

    for (int i = 0; i < num_steps; i++) {
        // Core range in audio samples (the part we keep)
        int core_start = i * audio_stride;
        int core_end = core_start + audio_stride;
        if (core_end > T_audio) core_end = T_audio;

        // Window with overlap context
        int win_start = core_start - audio_overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + audio_overlap;
        if (win_end > T_audio) win_end = T_audio;
        int win_len = win_end - win_start;

        // Encode this window
        int tile_T = vae_enc_compute(m, audio + win_start * 2, win_len);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE-Enc] FATAL: tile %d encode failed\n", i);
            return -1;
        }

        // Determine downsample factor from first tile
        if (i == 0) {
            downsample_factor = (float)tile_T / (float)win_len;
            fprintf(stderr, "[VAE-Enc] Downsample factor: %.6f (expected ~1/1920)\n",
                    downsample_factor);
        }

        // Trim in latent frames (mirror of decoder trim logic)
        int added_start = core_start - win_start;
        int trim_start = (int)roundf((float)added_start * downsample_factor);
        int added_end = win_end - core_end;
        int trim_end = (int)roundf((float)added_end * downsample_factor);

        int end_idx = (trim_end > 0) ? (tile_T - trim_end) : tile_T;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        if (latent_write_pos + core_len > max_T_latent) {
            fprintf(stderr, "[VAE-Enc] FATAL: tiled output exceeds max_T_latent\n");
            return -1;
        }

        // Read tile output [ne0=tile_T, ne1=128], extract mean (ch 0..63), transpose
        // Only read the first 64 channels (mean), skip scale channels 64..127
        size_t out_bytes = (size_t)128 * tile_T * sizeof(float);
        std::vector<float> raw(128 * tile_T);
        ggml_backend_tensor_get(m->graph_output, raw.data(), 0, out_bytes);

        for (int t = 0; t < core_len; t++)
            for (int c = 0; c < 64; c++)
                latent_out[(latent_write_pos + t) * 64 + c] =
                    raw[c * tile_T + (trim_start + t)];

        latent_write_pos += core_len;
    }

    fprintf(stderr, "[VAE-Enc] Tiled encode done: %d tiles -> T_latent=%d (%.2fs @ 48kHz)\n",
            num_steps, latent_write_pos, (float)T_audio / 48000.0f);

    return latent_write_pos;
}

// Free all resources
static void vae_enc_free(VAEEncoder * m) {
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

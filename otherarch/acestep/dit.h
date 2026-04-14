#pragma once
// dit.h: ACE-Step DiT (Diffusion Transformer) via ggml compute graph
// Ported from Python ACE-Step-1.5 reference. Same weights, loaded from GGUF.
//
// Architecture: 24-layer transformer with AdaLN, GQA self-attn + cross-attn, SwiGLU MLP.
// Flow matching: 8 Euler steps (turbo schedule).
//
// ggml ops used: rms_norm, mul_mat, rope_ext, flash_attn_ext, swiglu_split,
//                conv_transpose_1d, add, mul, scale, view, reshape, permute.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf_weights.h"
#include "backend.h"

#include "debug.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Config (mirrors dit.cuh DiTConfig)
struct DiTGGMLConfig {
    int hidden_size        = 2048;
    int intermediate_size  = 6144;
    int n_heads            = 16;
    int n_kv_heads         = 8;
    int head_dim           = 128;
    int n_layers           = 24;
    int in_channels        = 192;           // after context concat
    int out_channels       = 64;            // audio_acoustic_hidden_dim
    int patch_size         = 2;
    int sliding_window     = 128;
    float rope_theta       = 1000000.0f;
    float rms_norm_eps     = 1e-6f;
};

// Layer weights
struct DiTGGMLTembWeights {
    struct ggml_tensor * linear_1_w;        // [256, hidden]
    struct ggml_tensor * linear_1_b;        // [hidden]
    struct ggml_tensor * linear_2_w;        // [hidden, hidden]
    struct ggml_tensor * linear_2_b;        // [hidden]
    struct ggml_tensor * time_proj_w;       // [hidden, 6*hidden]
    struct ggml_tensor * time_proj_b;       // [6*hidden]
};

struct DiTGGMLLayer {
    // Self-attention
    struct ggml_tensor * self_attn_norm;    // [hidden]
    struct ggml_tensor * sa_qkv;            // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * sa_qk;             // [hidden, (Nh+Nkv)*D] partial QK fused (or NULL)
    struct ggml_tensor * sa_q_proj;         // separate fallback (NULL when any fusion active)
    struct ggml_tensor * sa_k_proj;
    struct ggml_tensor * sa_v_proj;
    struct ggml_tensor * sa_q_norm;         // [head_dim]
    struct ggml_tensor * sa_k_norm;         // [head_dim]
    struct ggml_tensor * sa_o_proj;         // [n_heads*head_dim, hidden]

    // Cross-attention
    struct ggml_tensor * cross_attn_norm;   // [hidden]
    struct ggml_tensor * ca_qkv;            // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * ca_q_proj;         // separate (always for cross-attn with mixed types)
    struct ggml_tensor * ca_kv;             // [hidden, 2*Nkv*D] fused KV (or NULL)
    struct ggml_tensor * ca_k_proj;
    struct ggml_tensor * ca_v_proj;
    struct ggml_tensor * ca_q_norm;         // [head_dim]
    struct ggml_tensor * ca_k_norm;         // [head_dim]
    struct ggml_tensor * ca_o_proj;         // [n_heads*head_dim, hidden]

    // MLP
    struct ggml_tensor * mlp_norm;          // [hidden]
    struct ggml_tensor * gate_up;           // [hidden, 2*intermediate] fused (or NULL)
    struct ggml_tensor * gate_proj;         // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * up_proj;           // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * down_proj;         // [intermediate, hidden]

    // AdaLN scale-shift table: [6*hidden] (6 rows of [hidden])
    struct ggml_tensor * scale_shift_table; // [hidden, 6] in ggml layout

    int layer_type;  // 0=sliding, 1=full
};

// Full model
#define DIT_GGML_MAX_LAYERS 32

struct DiTGGML {
    DiTGGMLConfig cfg;

    // Timestep embeddings
    DiTGGMLTembWeights time_embed;
    DiTGGMLTembWeights time_embed_r;

    // proj_in: Conv1d(in_channels, hidden, kernel=2, stride=2)
    struct ggml_tensor * proj_in_w;          // [in_ch*P, H] pre-permuted F32
    struct ggml_tensor * proj_in_b;          // [hidden]

    // condition_embedder: Linear(hidden, hidden)
    struct ggml_tensor * cond_emb_w;         // [hidden, hidden]
    struct ggml_tensor * cond_emb_b;         // [hidden]

    // Layers
    DiTGGMLLayer layers[DIT_GGML_MAX_LAYERS];

    // Output
    struct ggml_tensor * norm_out;           // [hidden]
    struct ggml_tensor * out_scale_shift;    // [hidden, 2] in ggml layout
    struct ggml_tensor * proj_out_w;         // [H, out_ch*P] pre-permuted+transposed F32
    struct ggml_tensor * proj_out_b;         // [out_channels]

    // CFG (classifier-free guidance, used by base/sft models)
    struct ggml_tensor * null_condition_emb; // [hidden] or NULL if not present

    // Backend
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    bool use_flash_attn;

    // Weight storage
    WeightCtx wctx;

    // Pre-allocated constant for AdaLN (1+scale) fusion
    struct ggml_tensor * scalar_one;  // [1] = 1.0f, broadcast in ggml_add
};

// Load timestep embedding weights
static void dit_ggml_load_temb(DiTGGMLTembWeights * w, WeightCtx * wctx,
                                const GGUFModel & gf, const std::string & prefix) {
    w->linear_1_w  = gf_load_tensor(wctx, gf, prefix + ".linear_1.weight");
    w->linear_1_b  = gf_load_tensor_f32(wctx, gf, prefix + ".linear_1.bias");
    w->linear_2_w  = gf_load_tensor(wctx, gf, prefix + ".linear_2.weight");
    w->linear_2_b  = gf_load_tensor_f32(wctx, gf, prefix + ".linear_2.bias");
    w->time_proj_w = gf_load_tensor(wctx, gf, prefix + ".time_proj.weight");
    w->time_proj_b = gf_load_tensor_f32(wctx, gf, prefix + ".time_proj.bias");
}

// Load proj_in weight: GGUF [H, in_ch, P] -> pre-permuted 2D [in_ch*P, H] F32
// Eliminates runtime permute+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_in_w(
        WeightCtx * wctx, const GGUFModel & gf, const std::string & name,
        int H, int in_ch, int P) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[GGUF] FATAL: meta tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    size_t offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw = gf.mapping + gf.data_offset + offset;

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_ch * P, H);
    ggml_set_name(dst, name.c_str());

    size_t n = (size_t)in_ch * P * H;
    wctx->staging.emplace_back(n);
    auto & buf = wctx->staging.back();

    // src ggml [P, in_ch, H]: elem(p, ic, h) = raw[h*P*in_ch + ic*P + p]
    // dst ggml [in_ch*P, H]:  elem(j, h)     = buf[h*in_ch*P + j]  where j = p*in_ch + ic
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++)
            for (int ic = 0; ic < in_ch; ic++)
                for (int p = 0; p < P; p++)
                    buf[h*in_ch*P + p*in_ch + ic] = read_fn(h*P*in_ch + ic*P + p);
    };
    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *)raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *)&s[i]); });
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src->type == GGML_TYPE_F32) {
        const float * s = (const float *)raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[GGUF] FATAL: unsupported type %d for '%s' in proj_in pre-permute\n",
                src->type, name.c_str());
        exit(1);
    }
    wctx->pending.push_back({dst, buf.data(), n * sizeof(float), 0});
    return dst;
}

// Load proj_out weight: GGUF [H, out_ch, P] -> pre-permuted+transposed 2D [H, out_ch*P] F32
// Eliminates runtime permute+cont+transpose+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_out_w(
        WeightCtx * wctx, const GGUFModel & gf, const std::string & name,
        int H, int out_ch, int P) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[GGUF] FATAL: meta tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    size_t offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw = gf.mapping + gf.data_offset + offset;

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, H, out_ch * P);
    ggml_set_name(dst, name.c_str());

    size_t n = (size_t)out_ch * P * H;
    wctx->staging.emplace_back(n);
    auto & buf = wctx->staging.back();

    // src ggml [P, out_ch, H]: elem(p, oc, h) = raw[h*P*out_ch + oc*P + p]
    // dst ggml [H, out_ch*P]:  elem(h, j)     = buf[j*H + h]  where j = p*out_ch + oc
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++)
            for (int oc = 0; oc < out_ch; oc++)
                for (int p = 0; p < P; p++)
                    buf[(p*out_ch + oc)*H + h] = read_fn(h*P*out_ch + oc*P + p);
    };
    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *)raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *)&s[i]); });
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src->type == GGML_TYPE_F32) {
        const float * s = (const float *)raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[GGUF] FATAL: unsupported type %d for '%s' in proj_out pre-permute\n",
                src->type, name.c_str());
        exit(1);
    }
    wctx->pending.push_back({dst, buf.data(), n * sizeof(float), 0});
    return dst;
}

// Load full DiT model from GGUF
static bool dit_ggml_load(DiTGGML * m, const char * gguf_path, DiTGGMLConfig cfg) {
    m->cfg = cfg;

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // Count tensors: temb(6*2) + proj_in(2) + cond_emb(2) + layers(19*24) + output(4) + null_cond(1) + scalar_one(1) = 476
    int n_tensors = 6 * 2 + 2 + 2 + 19 * cfg.n_layers + 4 + 1 + 1;
    wctx_init(&m->wctx, n_tensors);

    // Timestep embeddings
    dit_ggml_load_temb(&m->time_embed,  &m->wctx, gf, "decoder.time_embed");
    dit_ggml_load_temb(&m->time_embed_r, &m->wctx, gf, "decoder.time_embed_r");

    // proj_in: Conv1d weight [hidden, in_ch, patch_size]
    // Pre-permuted to 2D [in_ch*P, H] F32 at load time
    m->proj_in_w = dit_load_proj_in_w(&m->wctx, gf, "decoder.proj_in.1.weight",
                                       cfg.hidden_size, cfg.in_channels, cfg.patch_size);
    m->proj_in_b = gf_load_tensor_f32(&m->wctx, gf, "decoder.proj_in.1.bias");

    // condition_embedder
    m->cond_emb_w = gf_load_tensor(&m->wctx, gf, "decoder.condition_embedder.weight");
    m->cond_emb_b = gf_load_tensor_f32(&m->wctx, gf, "decoder.condition_embedder.bias");

    // Layers
    for (int i = 0; i < cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
        std::string p(prefix);
        DiTGGMLLayer & ly = m->layers[i];

        // Self-attention: try full QKV, partial QK, separate
        ly.self_attn_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn_norm.weight");
        ly.sa_qkv = gf_load_qkv_fused(&m->wctx, gf,
                        p + ".self_attn.q_proj.weight",
                        p + ".self_attn.k_proj.weight",
                        p + ".self_attn.v_proj.weight");
        if (!ly.sa_qkv) {
            // Try Q+K fusion (same input, often same type in K-quants)
            ly.sa_qk = gf_load_pair_fused(&m->wctx, gf,
                            p + ".self_attn.q_proj.weight",
                            p + ".self_attn.k_proj.weight");
            if (ly.sa_qk) {
                ly.sa_v_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.v_proj.weight");
                if (i == 0) fprintf(stderr, "[DiT] Self-attn: Q+K fused, V separate\n");
            } else {
                ly.sa_q_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.q_proj.weight");
                ly.sa_k_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.k_proj.weight");
                ly.sa_v_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.v_proj.weight");
                if (i == 0) fprintf(stderr, "[DiT] Self-attn: all separate (3 types differ)\n");
            }
        } else {
            if (i == 0) fprintf(stderr, "[DiT] Self-attn: Q+K+V fused\n");
        }
        ly.sa_q_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.o_proj.weight");

        // Cross-attention: try full QKV, K+V fused, separate
        ly.cross_attn_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn_norm.weight");
        ly.ca_qkv = gf_load_qkv_fused(&m->wctx, gf,
                        p + ".cross_attn.q_proj.weight",
                        p + ".cross_attn.k_proj.weight",
                        p + ".cross_attn.v_proj.weight");
        if (!ly.ca_qkv) {
            ly.ca_q_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.q_proj.weight");
            // Try K+V fusion (same input enc, may share type)
            ly.ca_kv = gf_load_pair_fused(&m->wctx, gf,
                            p + ".cross_attn.k_proj.weight",
                            p + ".cross_attn.v_proj.weight");
            if (ly.ca_kv) {
                if (i == 0) fprintf(stderr, "[DiT] Cross-attn: Q separate, K+V fused\n");
            } else {
                ly.ca_k_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.k_proj.weight");
                ly.ca_v_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.v_proj.weight");
                if (i == 0) fprintf(stderr, "[DiT] Cross-attn: all separate\n");
            }
        } else {
            if (i == 0) fprintf(stderr, "[DiT] Cross-attn: Q+K+V fused\n");
        }
        ly.ca_q_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.o_proj.weight");

        // MLP: try gate+up fusion (same input, same pattern as QKV)
        ly.mlp_norm  = gf_load_tensor_f32(&m->wctx, gf, p + ".mlp_norm.weight");
        ly.gate_up = gf_load_pair_fused(&m->wctx, gf,
                        p + ".mlp.gate_proj.weight",
                        p + ".mlp.up_proj.weight");
        if (ly.gate_up) {
            if (i == 0) fprintf(stderr, "[DiT] MLP: gate+up fused\n");
        } else {
            ly.gate_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.gate_proj.weight");
            ly.up_proj   = gf_load_tensor(&m->wctx, gf, p + ".mlp.up_proj.weight");
            if (i == 0) fprintf(stderr, "[DiT] MLP: gate+up separate (types differ)\n");
        }
        ly.down_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.down_proj.weight");

        // AdaLN scale_shift_table [1, 6, hidden] in GGUF
        ly.scale_shift_table = gf_load_tensor_f32(&m->wctx, gf, p + ".scale_shift_table");

        ly.layer_type = (i % 2 == 0) ? 0 : 1;  // 0=sliding, 1=full
    }

    // Output
    m->norm_out        = gf_load_tensor_f32(&m->wctx, gf, "decoder.norm_out.weight");
    m->out_scale_shift = gf_load_tensor_f32(&m->wctx, gf, "decoder.scale_shift_table");
    m->proj_out_w      = dit_load_proj_out_w(&m->wctx, gf, "decoder.proj_out.1.weight",
                                              cfg.hidden_size, cfg.out_channels, cfg.patch_size);
    m->proj_out_b      = gf_load_tensor_f32(&m->wctx, gf, "decoder.proj_out.1.bias");

    // Null condition embedding for CFG (base/sft models; turbo has it but unused at inference)
    m->null_condition_emb = gf_try_load_tensor(&m->wctx, gf, "null_condition_emb");
    if (m->null_condition_emb) {
        fprintf(stderr, "[Load] null_condition_emb found (CFG available)\n");
    }

    // Scalar constant for AdaLN (1+scale) fusion
    static const float one_val = 1.0f;
    m->scalar_one = ggml_new_tensor_1d(m->wctx.ctx, GGML_TYPE_F32, 1);
    m->wctx.pending.push_back({m->scalar_one, &one_val, sizeof(float), 0});

    // Allocate backend buffer and copy weights
    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[Load] DiT: %d layers, H=%d, Nh=%d/%d, D=%d\n",
            cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    return true;
}

// Backend init
static void dit_ggml_init_backend(DiTGGML * m) {
    BackendPair bp = backend_init("DiT");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
    // flash_attn_ext accumulates in F16 on CPU, causing audible drift over
    // 24 layers x 8 steps. Use F32 manual attention on CPU instead.
    m->use_flash_attn = (bp.backend != bp.cpu_backend);
}

// Graph builder: single DiT layer (self-attention block)
// Incremental approach: build and validate one block at a time.
//
// ggml tensor layout reminder:
//   [S, H] in math = ne[0]=H, ne[1]=S in ggml
//   [Nh, S, D] in math = ne[0]=D, ne[1]=S, ne[2]=Nh in ggml

// Helper: ensure tensor is f32 (cast if bf16/f16)
static struct ggml_tensor * dit_ggml_f32(
        struct ggml_context * ctx,
        struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Helper: RMSNorm + weight multiply
static struct ggml_tensor * dit_ggml_rms_norm_weighted(
        struct ggml_context * ctx,
        struct ggml_tensor * x,         // [H, S]
        struct ggml_tensor * weight,    // [H]
        float eps) {
    struct ggml_tensor * norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, dit_ggml_f32(ctx, weight));
}

// Helper: Linear layer (no bias)
// weight: [in, out] in ggml (= [out, in] in PyTorch)
// input:  [in, S]
// output: [out, S]
static struct ggml_tensor * dit_ggml_linear(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        struct ggml_tensor * input) {
    return ggml_mul_mat(ctx, weight, input);
}

// Helper: Linear layer with bias
static struct ggml_tensor * dit_ggml_linear_bias(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        struct ggml_tensor * bias,
        struct ggml_tensor * input) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    return ggml_add(ctx, out, dit_ggml_f32(ctx, bias));
}

// Helper: AdaLN modulate
// out = norm * (1 + scale) + shift
// norm: [H, S], scale: [H], shift: [H]
static struct ggml_tensor * dit_ggml_adaln(
        struct ggml_context * ctx,
        struct ggml_tensor * norm,
        struct ggml_tensor * scale,
        struct ggml_tensor * shift,
        struct ggml_tensor * one) {
    // norm * (1 + scale) + shift
    // one is [1] = 1.0, broadcasts to [H]; avoids expensive [H,S,N] add
    struct ggml_tensor * one_plus_s = ggml_add(ctx, scale, one);   // [H] + [1] -> [H]
    struct ggml_tensor * scaled = ggml_mul(ctx, norm, one_plus_s); // [H,S,N]
    return ggml_add(ctx, scaled, shift);                           // [H,S,N]
}

// Helper: Gated residual
// out = residual + x * gate
// residual: [H, S], x: [H, S], gate: [H]
// NOTE: no sigmoid, gate is a raw scaling factor (matches Python reference)
static struct ggml_tensor * dit_ggml_gated_add(
        struct ggml_context * ctx,
        struct ggml_tensor * residual,
        struct ggml_tensor * x,
        struct ggml_tensor * gate) {
    struct ggml_tensor * gated = ggml_mul(ctx, x, gate); // broadcast [H] over [H,S]
    return ggml_add(ctx, residual, gated);
}

// Build timestep embedding subgraph
// t_scalar: [1] f32, returns temb [H] and *out_tproj [6H]
// suffix: "_t" or "_r" for naming intermediate tensors
static struct ggml_tensor * dit_ggml_build_temb(
        struct ggml_context * ctx,
        DiTGGMLTembWeights * w,
        struct ggml_tensor * t_scalar,
        struct ggml_tensor ** out_tproj,
        const char * suffix = "") {

    // scale timestep by 1000 (diffusion convention, matches Python)
    struct ggml_tensor * t_scaled = ggml_scale(ctx, t_scalar, 1000.0f);

    // sinusoidal embedding: [1] -> [256]
    struct ggml_tensor * sinusoidal = ggml_timestep_embedding(ctx, t_scaled, 256, 10000);
    {
        char name[64];
        snprintf(name, sizeof(name), "sinusoidal%s", suffix);
        ggml_set_name(sinusoidal, name);
        ggml_set_output(sinusoidal);
    }

    // linear1 + silu: [256] -> [H]
    struct ggml_tensor * h = dit_ggml_linear_bias(ctx, w->linear_1_w, w->linear_1_b, sinusoidal);
    {
        char name[64];
        snprintf(name, sizeof(name), "temb_lin1%s", suffix);
        ggml_set_name(h, name);
        ggml_set_output(h);
    }
    h = ggml_silu(ctx, h);

    // linear2: [H] -> [H]
    struct ggml_tensor * temb = dit_ggml_linear_bias(ctx, w->linear_2_w, w->linear_2_b, h);

    // silu + proj: [H] -> [6H]
    struct ggml_tensor * h2 = ggml_silu(ctx, temb);
    *out_tproj = dit_ggml_linear_bias(ctx, w->time_proj_w, w->time_proj_b, h2);

    return temb;  // [H] (used for output adaln)
}

// F32 manual attention (fallback when flash_attn_ext is not available or imprecise).
// Q: [D, S, Nh], K: [D, S_kv, Nkv], V: [D, S_kv, Nkv]
// mask: [S_kv, S] F16 or NULL, scale: 1/sqrt(D)
// Returns: [D, Nh, S] (same layout as flash_attn_ext output)
static struct ggml_tensor * dit_attn_f32(
        struct ggml_context * ctx,
        struct ggml_tensor * q,
        struct ggml_tensor * k,
        struct ggml_tensor * v,
        struct ggml_tensor * mask,
        float scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Build self-attention sub-graph for a single layer.
// norm_sa: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N] (self-attention output, NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_self_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_sa,   // [H, S, N] pre-normalized + AdaLN-modulated
        struct ggml_tensor * positions, // [S*N] int32 position indices for RoPE
        struct ggml_tensor * mask,      // [S, S] or NULL (sliding window mask)
        int S, int N, int layer_idx = -1) {

    DiTGGMLConfig & c = m->cfg;
    int D  = c.head_dim;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;

    // 1) QKV projections (full fused, QK partial, separate)
    struct ggml_tensor * q, * k, * v;
    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;
    if (ly->sa_qkv) {
        struct ggml_tensor * qkv = dit_ggml_linear(ctx, ly->sa_qkv, norm_sa);
        q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, q_dim, S, N, qkv->nb[1], qkv->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t)q_dim * qkv->nb[0]));
        v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t)(q_dim + kv_dim) * qkv->nb[0]));
    } else if (ly->sa_qk) {
        struct ggml_tensor * qk = dit_ggml_linear(ctx, ly->sa_qk, norm_sa);
        q = ggml_cont(ctx, ggml_view_3d(ctx, qk, q_dim, S, N, qk->nb[1], qk->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qk, kv_dim, S, N, qk->nb[1], qk->nb[2], (size_t)q_dim * qk->nb[0]));
        v = dit_ggml_linear(ctx, ly->sa_v_proj, norm_sa);
    } else {
        q = dit_ggml_linear(ctx, ly->sa_q_proj, norm_sa);
        k = dit_ggml_linear(ctx, ly->sa_k_proj, norm_sa);
        v = dit_ggml_linear(ctx, ly->sa_v_proj, norm_sa);
    }

    // 2) Reshape to heads: [Nh*D, S, N] -> [D, Nh, S, N]
    //    Rope merges S*N then restores 4D. Permute to flash_attn layout after rope.
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, N);

    // 4) QK-Norm: per-head RMSNorm on D dimension
    //    [D, Nh, S] rms_norm operates on ne[0]=D
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->sa_k_norm));

    // 5) RoPE (bidirectional, sequential positions)
    //    ggml_rope_ext asserts ne[2] == positions.ne[0].
    //    With batch N>1, positions has S*N elements (repeated [0..S-1] per batch).
    //    Merge S and N before rope, then restore 4D after.
    q = ggml_reshape_3d(ctx, q, D, Nh, S * N);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S * N);
    q = ggml_rope_ext(ctx, q, positions, NULL,
                       D, 2 /*mode=NEOX*/, 0 /*n_ctx_orig*/,
                       c.rope_theta, 1.0f /*freq_scale*/,
                       0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL,
                       D, 2, 0,
                       c.rope_theta, 1.0f,
                       0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);

    if (layer_idx == 0) {
        ggml_set_name(q, "layer0_q_after_rope");
        ggml_set_output(q);
        ggml_set_name(k, "layer0_k_after_rope");
        ggml_set_output(k);
    }

    // 6) Permute for flash_attn_ext: [D, Nh, S, N] -> [D, S, Nh, N]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // 7) Attention (flash on GPU, F32 manual on CPU)
    //    Q[D, S, Nh, N], K[D, S, Nkv, N], V[D, S, Nkv, N]
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = m->use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f)
        : dit_attn_f32(ctx, q, k, v, mask, scale);

    // Both return [D, Nh, S, N]
    // Reshape: [D, Nh, S, N] -> [D*Nh, S, N] = [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    if (layer_idx == 0) {
        ggml_set_name(attn, "layer0_attn_out");
        ggml_set_output(attn);
    }

    // 8) O projection: [Nh*D, S, N] -> [H, S, N]
    struct ggml_tensor * out = dit_ggml_linear(ctx, ly->sa_o_proj, attn);
    return out;
}

// Build MLP sub-graph: SwiGLU
// norm_ffn: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N]
static struct ggml_tensor * dit_ggml_build_mlp(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ffn,
        int S) {

    struct ggml_tensor * ff;
    if (ly->gate_up) {
        // Fused: single matmul [H, 2*I] x [H, S, N] -> [2*I, S, N], then swiglu splits ne[0]
        struct ggml_tensor * gu = dit_ggml_linear(ctx, ly->gate_up, norm_ffn);
        ff = ggml_swiglu(ctx, gu);
    } else {
        // Separate: two matmuls + split swiglu
        struct ggml_tensor * gate = dit_ggml_linear(ctx, ly->gate_proj, norm_ffn);
        struct ggml_tensor * up   = dit_ggml_linear(ctx, ly->up_proj, norm_ffn);
        ff = ggml_swiglu_split(ctx, gate, up);
    }

    // Down projection: [I, S] -> [H, S]
    return dit_ggml_linear(ctx, ly->down_proj, ff);
}

// Build cross-attention sub-graph for a single layer.
// norm_ca: [H, S, N] pre-normalized hidden state (Q source)
// enc:     [H, enc_S, N] condition-embedded encoder states (K/V source)
// Returns: output [H, S, N] (NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_cross_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ca,   // [H, S, N]
        struct ggml_tensor * enc,       // [H, enc_S, N]
        struct ggml_tensor * positions, // unused, kept for consistency
        int S, int enc_S, int N) {

    DiTGGMLConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    (void)positions;  // cross-attn has no RoPE

    // Q from hidden, KV from encoder (full fused, Q+KV partial, separate)
    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;
    struct ggml_tensor * q, * k, * v;
    if (ly->ca_qkv) {
        // Full QKV fused: split Q from hidden, KV from enc via weight views
        struct ggml_tensor * w_q  = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], q_dim,
                                                  ly->ca_qkv->nb[1], 0);
        struct ggml_tensor * w_kv = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], 2 * kv_dim,
                                                  ly->ca_qkv->nb[1], (size_t)q_dim * ly->ca_qkv->nb[1]);
        q = ggml_mul_mat(ctx, w_q, norm_ca);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, w_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t)kv_dim * kv->nb[0]));
    } else if (ly->ca_kv) {
        // Q separate, K+V fused
        q = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, ly->ca_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t)kv_dim * kv->nb[0]));
    } else {
        q = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca);
        k = dit_ggml_linear(ctx, ly->ca_k_proj, enc);
        v = dit_ggml_linear(ctx, ly->ca_v_proj, enc);
    }

    // reshape to [D, heads, seq, N] then permute to [D, seq, heads, N]
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [D, S, Nh, N]

    k = ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [D, enc_S, Nkv, N]

    v = ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);   // [D, enc_S, Nkv, N]

    // QK-norm (per head)
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->ca_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->ca_k_norm));

    // no RoPE for cross-attention
    // no mask (attend to all encoder positions)
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = m->use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f)
        : dit_attn_f32(ctx, q, k, v, NULL, scale);

    // Attention output: [D, Nh, S, N], reshape to [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    // O projection
    return dit_ggml_linear(ctx, ly->ca_o_proj, attn);
}

// Build one full DiT layer (AdaLN + self-attn + cross-attn + FFN + gated residuals)
// hidden: [H, S, N], tproj: [6H] (combined timestep projection)
// enc: [H, enc_S, N] (condition-embedded encoder states, or NULL to skip cross-attn)
// sw_mask: [S, S] sliding window mask (or NULL for full attention)
// Returns: updated hidden [H, S, N]
static struct ggml_tensor * dit_ggml_build_layer(
        struct ggml_context * ctx,
        DiTGGML * m,
        int layer_idx,
        struct ggml_tensor * hidden,    // [H, S, N]
        struct ggml_tensor * tproj,     // [6H] f32 combined temb projection
        struct ggml_tensor * enc,       // [H, enc_S, N] or NULL
        struct ggml_tensor * positions, // [S] int32
        struct ggml_tensor * sw_mask,   // [S, S] or NULL
        int S, int enc_S, int N) {

    DiTGGMLConfig & c = m->cfg;
    DiTGGMLLayer * ly = &m->layers[layer_idx];
    int H = c.hidden_size;

    // AdaLN: scale_shift_table [6, H] + tproj [6H] -> 6 vectors of [H]
    // scale_shift_table is stored as bf16, cast to f32 for arithmetic
    struct ggml_tensor * ss = ly->scale_shift_table;
    if (ss->type != GGML_TYPE_F32) {
        ss = ggml_cast(ctx, ss, GGML_TYPE_F32);
    }
    // flatten [H, 6] -> [6H] (ggml ne[0]=H, ne[1]=6, contiguous = 6H floats)
    struct ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ss, 6 * H);
    struct ggml_tensor * adaln = ggml_add(ctx, ss_flat, tproj);  // [6H] f32

    // extract 6 modulation vectors [H] each
    size_t Hb = H * sizeof(float);
    struct ggml_tensor * shift_sa  = ggml_view_1d(ctx, adaln, H, 0 * Hb);
    struct ggml_tensor * scale_sa  = ggml_view_1d(ctx, adaln, H, 1 * Hb);
    struct ggml_tensor * gate_sa   = ggml_view_1d(ctx, adaln, H, 2 * Hb);
    struct ggml_tensor * shift_ffn = ggml_view_1d(ctx, adaln, H, 3 * Hb);
    struct ggml_tensor * scale_ffn = ggml_view_1d(ctx, adaln, H, 4 * Hb);
    struct ggml_tensor * gate_ffn  = ggml_view_1d(ctx, adaln, H, 5 * Hb);

    // Self-attention with AdaLN + gated residual
    struct ggml_tensor * residual = hidden;
    struct ggml_tensor * norm_sa = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    norm_sa = dit_ggml_adaln(ctx, norm_sa, scale_sa, shift_sa, m->scalar_one);

    if (layer_idx == 0) {
        ggml_set_name(norm_sa, "layer0_sa_input");
        ggml_set_output(norm_sa);
    }

    // select mask: even layers use sliding window, odd layers use full attention
    struct ggml_tensor * mask = (ly->layer_type == 0) ? sw_mask : NULL;
    struct ggml_tensor * sa_out = dit_ggml_build_self_attn(ctx, m, ly, norm_sa, positions, mask, S, N, layer_idx);

    if (layer_idx == 0) {
        ggml_set_name(sa_out, "layer0_sa_output");
        ggml_set_output(sa_out);
    }

    hidden = dit_ggml_gated_add(ctx, residual, sa_out, gate_sa);

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_self_attn");
        ggml_set_output(hidden);
    }

    // Cross-attention (no gate, simple residual add)
    if (enc) {
        struct ggml_tensor * norm_ca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
        struct ggml_tensor * ca_out = dit_ggml_build_cross_attn(ctx, m, ly, norm_ca, enc, positions, S, enc_S, N);
        hidden = ggml_add(ctx, hidden, ca_out);
    }

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_cross_attn");
        ggml_set_output(hidden);
    }

    // FFN with AdaLN + gated residual
    residual = hidden;
    struct ggml_tensor * norm_ffn = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn = dit_ggml_adaln(ctx, norm_ffn, scale_ffn, shift_ffn, m->scalar_one);
    struct ggml_tensor * ffn_out = dit_ggml_build_mlp(ctx, m, ly, norm_ffn, S);
    hidden = dit_ggml_gated_add(ctx, residual, ffn_out, gate_ffn);

    return hidden;
}

// Build the full DiT forward graph (all layers).
// Returns the final output tensor (velocity prediction).
// N = batch size (number of samples to denoise in parallel).
//
// Graph inputs (ggml [ne0, ne1, ne2] notation):
//   "input_latents"   [in_channels, T, N]  concat(context_latents, xt) per sample
//   "enc_hidden"      [H, enc_S, N]        text encoder hidden states (N copies)
//   "t"               [1] f32              flow matching timestep (shared)
//   "t_r"             [1] f32              reference timestep (shared)
//   "positions"       [S*N] i32            position indices 0..S-1 repeated N times
//   "sw_mask"         [S, S, 1, N] f16     sliding window mask (N identical copies)
//
// Graph outputs:
//   "velocity"        [out_channels, T, N]  predicted flow velocity
static struct ggml_cgraph * dit_ggml_build_graph(
        DiTGGML * m,
        struct ggml_context * ctx,
        int T,                 // temporal length (before patching)
        int enc_S,             // encoder sequence length
        int N,                 // batch size
        struct ggml_tensor ** p_input,     // [out] input tensor to fill
        struct ggml_tensor ** p_output) {  // [out] output tensor to read

    DiTGGMLConfig & c = m->cfg;
    int S = T / c.patch_size;  // sequence length after patching
    int H = c.hidden_size;
    int P = c.patch_size;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Inputs

    // Concatenated latent: [in_channels, T, N] per sample
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, T, N);
    ggml_set_name(input, "input_latents");
    ggml_set_input(input);
    *p_input = input;

    // Encoder hidden states: [H, enc_S, N]
    struct ggml_tensor * enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, enc_S, N);
    ggml_set_name(enc_hidden, "enc_hidden");
    ggml_set_input(enc_hidden);

    // Timesteps: scalars
    struct ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_val, "t");
    ggml_set_input(t_val);

    struct ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tr_val, "t_r");
    ggml_set_input(tr_val);

    // Position indices for RoPE: [N*S] with values [0..S-1] repeated N times.
    // The CUDA rope kernel indexes positions by channel_x = row / ne1 which
    // linearizes (ne2, ne3) = (S, N). Batch b reads pos[b*S + s], so we must
    // repeat the sequence for each batch element.
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S * N);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ggml pitfall: flash_attn_ext reads mask as fp16!
    // Must be 4D [S, S, 1, N] not 2D [S, S]: the CUDA flash_attn_mask_to_KV_max
    // optimization kernel offsets the mask pointer by sequence*nb[3] per batch element.
    // With 2D mask (ne[3]=1), batch 1+ reads out of bounds. Replicate mask N times.
    struct ggml_tensor * sw_mask = NULL;
    if (c.sliding_window > 0 && S > c.sliding_window) {
        sw_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
        ggml_set_name(sw_mask, "sw_mask");
        ggml_set_input(sw_mask);
    }

    // 1) Timestep embeddings
    struct ggml_tensor * tproj;
    struct ggml_tensor * temb;

    {
        struct ggml_tensor * tproj_t;
        struct ggml_tensor * temb_t = dit_ggml_build_temb(ctx, &m->time_embed, t_val, &tproj_t, "_t");
        ggml_set_name(temb_t, "temb_t");
        ggml_set_output(temb_t);

        struct ggml_tensor * tproj_r;
        // Python passes (t - t_r) to time_embed_r, not t_r directly
        // In turbo mode t = t_r, so input is 0
        struct ggml_tensor * t_diff = ggml_sub(ctx, t_val, tr_val);
        struct ggml_tensor * temb_r = dit_ggml_build_temb(ctx, &m->time_embed_r, t_diff, &tproj_r, "_r");
        ggml_set_name(temb_r, "temb_r");
        ggml_set_output(temb_r);

        // combine: temb = temb_t + temb_r [H], tproj = tproj_t + tproj_r [6H]
        temb  = ggml_add(ctx, temb_t, temb_r);
        ggml_set_name(temb, "temb");
        ggml_set_output(temb);
        tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_name(tproj, "tproj");
        ggml_set_output(tproj);
    }

    // 2) proj_in: patchify + linear (weight pre-permuted at load time)
    ggml_set_name(input, "proj_in_input");
    ggml_set_output(input);
    struct ggml_tensor * patched = ggml_reshape_3d(ctx, input, c.in_channels * P, S, N);
    struct ggml_tensor * hidden = dit_ggml_linear_bias(ctx, m->proj_in_w, m->proj_in_b, patched);
    ggml_set_name(hidden, "hidden_after_proj_in");
    ggml_set_output(hidden);

    // 3) Condition embedder: project encoder hidden states
    struct ggml_tensor * enc = dit_ggml_linear_bias(ctx, m->cond_emb_w, m->cond_emb_b, enc_hidden);
    ggml_set_name(enc, "enc_after_cond_emb");
    ggml_set_output(enc);

    // 4) Transformer layers
    for (int i = 0; i < c.n_layers; i++) {
        hidden = dit_ggml_build_layer(ctx, m, i, hidden, tproj, enc, positions, sw_mask, S, enc_S, N);
        // Debug dumps at key layers: 0, 6, 12, 18, 23
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == c.n_layers - 1) {
            char lname[64];
            snprintf(lname, sizeof(lname), "hidden_after_layer%d", i);
            ggml_set_name(hidden, lname);
            ggml_set_output(hidden);
        }
    }

    // 5) Output: AdaLN + proj_out
    // out_scale_shift: [H, 2] -> cast to f32 if bf16, flatten to [2H]
    struct ggml_tensor * oss = m->out_scale_shift;
    if (oss->type != GGML_TYPE_F32) {
        oss = ggml_cast(ctx, oss, GGML_TYPE_F32);
    }
    struct ggml_tensor * oss_flat = ggml_reshape_1d(ctx, oss, 2 * H);

    size_t Hb = H * sizeof(float);
    struct ggml_tensor * out_shift = ggml_view_1d(ctx, oss_flat, H, 0);
    struct ggml_tensor * out_scale = ggml_view_1d(ctx, oss_flat, H, Hb);
    out_shift = ggml_add(ctx, out_shift, temb);
    out_scale = ggml_add(ctx, out_scale, temb);

    struct ggml_tensor * norm_out = dit_ggml_rms_norm_weighted(ctx, hidden, m->norm_out, c.rms_norm_eps);
    norm_out = dit_ggml_adaln(ctx, norm_out, out_scale, out_shift, m->scalar_one);

    // proj_out: weight pre-permuted+transposed at load time to [H, out_ch*P] F32
    struct ggml_tensor * output = dit_ggml_linear_bias(ctx, m->proj_out_w, m->proj_out_b, norm_out);
    output = ggml_reshape_3d(ctx, output, c.out_channels, T, N);

    ggml_set_name(output, "velocity");
    ggml_set_output(output);
    *p_output = output;

    ggml_build_forward_expand(gf, output);

    return gf;
}

// APG (Adaptive Projected Guidance) for DiT CFG
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

struct APGMomentumBuffer {
    double momentum;
    std::vector<double> running_average;
    bool initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++)
                running_average[i] = values[i] + momentum * running_average[i];
        }
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(
        const double * v0, const double * v1,
        double * out_par, double * out_orth,
        int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++)
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++)
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);

        for (int t = 0; t < T; t++) {
            int idx = t * Oc + c;
            double v1n = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward() exactly:
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
// Internal computation in double precision (Python uses .double()).
static void apg_forward(
        const float * pred_cond, const float * pred_uncond,
        float guidance_scale, APGMomentumBuffer & mbuf,
        float * result, int Oc, int T,
        float norm_threshold = 2.5f) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    std::vector<double> diff(n);
    for (int i = 0; i < n; i++)
        diff[i] = (double)pred_cond[i] - (double)pred_uncond[i];

    // 2. momentum update, then use smoothed diff
    mbuf.update(diff.data(), n);
    memcpy(diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++)
                norm2 += diff[t * Oc + c] * diff[t * Oc + c];
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s = (norm > 1e-60) ? fmin(1.0, (double)norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++)
                    diff[t * Oc + c] *= s;
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    std::vector<double> pred_cond_d(n), par(n), orth(n);
    for (int i = 0; i < n; i++) pred_cond_d[i] = (double)pred_cond[i];
    apg_project(diff.data(), pred_cond_d.data(), par.data(), orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double)guidance_scale - 1.0;
    for (int i = 0; i < n; i++)
        result[i] = (float)((double)pred_cond[i] + w * orth[i]);
}

// Flow matching generation loop (batched)
// Runs num_steps euler steps to denoise N latent samples in parallel.
//
// noise:            [N * T * Oc]  N contiguous [T, Oc] noise blocks
// context_latents:  [N * T * ctx_ch]  N contiguous context blocks
// enc_hidden:       [enc_S * H]  SINGLE encoder output (shared, will be broadcast to N)
// schedule:         array of num_steps timestep values
// output:           [N * T * Oc]  generated latents (caller-allocated)
static void dit_ggml_generate(
        DiTGGML * model,
        const float * noise,
        const float * context_latents,
        const float * enc_hidden_data,
        int enc_S,
        int T,
        int N,
        int num_steps,
        const float * schedule,
        float * output,
        float guidance_scale = 1.0f,
        const DebugDumper * dbg = nullptr,
        const float * context_switch = nullptr,
        int cover_steps = -1) {

    DiTGGMLConfig & c = model->cfg;
    int Oc    = c.out_channels;      // 64
    int ctx_ch = c.in_channels - Oc; // 128
    int in_ch = c.in_channels;       // 192
    int S     = T / c.patch_size;
    int n_per = T * Oc;              // elements per sample
    int n_total = N * n_per;         // total output elements
    int H     = c.hidden_size;

    fprintf(stderr, "[DiT] Batch N=%d, T=%d, S=%d, enc_S=%d\n", N, T, S, enc_S);

    // Graph context (generous fixed allocation, shapes are constant across steps)
    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_tensor * t_input  = NULL;
    struct ggml_tensor * t_output = NULL;
    struct ggml_cgraph * gf = dit_ggml_build_graph(model, ctx, T, enc_S, N,
                                                    &t_input, &t_output);

    fprintf(stderr, "[DiT] Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    struct ggml_tensor * t_enc = ggml_graph_get_tensor(gf, "enc_hidden");

    // Allocate compute buffers.
    // Critical: reset FIRST (clears old state), THEN force inputs to GPU, THEN alloc.
    // Without GPU forcing, inputs default to CPU where the scheduler aliases their
    // buffers with intermediates. enc_hidden is read at every cross-attn layer (24x),
    // so CPU aliasing corrupts it mid-graph. With N>1 the larger buffers trigger
    // more aggressive aliasing, causing batch sample 1+ to produce noise.
    ggml_backend_sched_reset(model->sched);
    if (model->backend != model->cpu_backend) {
        const char * input_names[] = {"enc_hidden", "input_latents", "t", "t_r", "positions", "sw_mask"};
        for (const char * iname : input_names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, iname);
            if (t) ggml_backend_sched_set_tensor_backend(model->sched, t, model->backend);
        }
    }
    if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
        fprintf(stderr, "FATAL: failed to allocate graph\n");
        ggml_free(ctx);
        return;
    }

    // Encoder hidden states: upload once (re-uploaded per step only when CFG swaps to null)
    // t_enc was declared above for backend forcing

    // t_r is set per-step in the loop (= t_curr, same as Python reference)
    struct ggml_tensor * t_tr = ggml_graph_get_tensor(gf, "t_r");

    // Positions: [0, 1, ..., S-1] repeated N times for batch rope indexing
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(S * N);
    for (int b = 0; b < N; b++)
        for (int i = 0; i < S; i++)
            pos_data[b * S + i] = i;
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));

    // Sliding window mask: [S, S, 1, N] fp16 - N identical copies
    struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, "sw_mask");
    std::vector<uint16_t> mask_data;
    if (t_mask) {
        int win = c.sliding_window;
        mask_data.resize(S * S * N);
        // fill first copy
        for (int qi = 0; qi < S; qi++)
            for (int ki = 0; ki < S; ki++) {
                int dist = (qi > ki) ? (qi - ki) : (ki - qi);
                float v = (dist <= win) ? 0.0f : -INFINITY;
                mask_data[ki * S + qi] = ggml_fp32_to_fp16(v);
            }
        // replicate for batch elements 1..N-1
        for (int b = 1; b < N; b++)
            memcpy(mask_data.data() + b * S * S, mask_data.data(), S * S * sizeof(uint16_t));
        ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));
    }

    // CFG setup
    bool do_cfg = guidance_scale > 1.0f;
    std::vector<float> null_enc_buf;
    std::vector<APGMomentumBuffer> apg_mbufs;

    if (do_cfg) {
        if (!model->null_condition_emb) {
            fprintf(stderr, "[DiT] WARNING: guidance_scale=%.1f but null_condition_emb not found. Disabling CFG.\n", guidance_scale);
            do_cfg = false;
        } else {
            int emb_n = (int)ggml_nelements(model->null_condition_emb);
            std::vector<float> null_emb(emb_n);

            if (model->null_condition_emb->type == GGML_TYPE_BF16) {
                std::vector<uint16_t> bf16_buf(emb_n);
                ggml_backend_tensor_get(model->null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
                for (int i = 0; i < emb_n; i++) {
                    uint32_t w = (uint32_t)bf16_buf[i] << 16;
                    memcpy(&null_emb[i], &w, 4);
                }
            } else {
                ggml_backend_tensor_get(model->null_condition_emb, null_emb.data(), 0, emb_n * sizeof(float));
            }

            // Broadcast [H] to [enc_S, H] then to N copies [H, enc_S, N]
            std::vector<float> null_enc_single(H * enc_S);
            for (int s = 0; s < enc_S; s++)
                memcpy(&null_enc_single[s * H], null_emb.data(), H * sizeof(float));
            null_enc_buf.resize(H * enc_S * N);
            for (int b = 0; b < N; b++)
                memcpy(null_enc_buf.data() + b * enc_S * H, null_enc_single.data(), enc_S * H * sizeof(float));

            if (dbg && dbg->enabled) {
                debug_dump_1d(dbg, "null_condition_emb", null_emb.data(), emb_n);
                debug_dump_2d(dbg, "null_enc_hidden", null_enc_single.data(), enc_S, H);
            }

            apg_mbufs.resize(N);

            fprintf(stderr, "[DiT] CFG enabled: guidance_scale=%.1f, 2x forward per step, N=%d\n", guidance_scale, N);
        }
    }

    // Prepare host buffers (all N samples contiguous)
    std::vector<float> xt(noise, noise + n_total);
    std::vector<float> vt(n_total);

    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
    }

    // input_buf: [in_ch, T, N] - pre-fill context_latents (constant across all steps)
    std::vector<float> input_buf(in_ch * T * N);
    for (int b = 0; b < N; b++)
        for (int t = 0; t < T; t++)
            memcpy(&input_buf[b * T * in_ch + t * in_ch],
                   &context_latents[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));

    // Pre-allocate enc_buf once (avoids heap alloc per step)
    std::vector<float> enc_buf(H * enc_S * N);
    for (int b = 0; b < N; b++)
        memcpy(enc_buf.data() + b * enc_S * H, enc_hidden_data, enc_S * H * sizeof(float));
    ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));

    struct ggml_tensor * t_t = ggml_graph_get_tensor(gf, "t");

    // Flow matching loop
    bool switched_cover = false;
    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];

         // Cover mode: switch context from cover to non-cover at cover_steps
        if (context_switch && cover_steps >= 0 && step >= cover_steps && !switched_cover) {
            switched_cover = true;
            for (int b = 0; b < N; b++)
                for (int t = 0; t < T; t++)
                    memcpy(&input_buf[b * T * in_ch + t * in_ch],
                           &context_switch[b * T * ctx_ch + t * ctx_ch],
                           ctx_ch * sizeof(float));
            fprintf(stderr, "[DiT] Cover: switched to non-cover context at step %d/%d\n",
                    step, num_steps);
        }

        // Set timestep (changes each step)
        if (t_t) {
            ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
        }
        if (t_tr) {
            ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
        }

        // Re-upload constants (scheduler may reuse input buffers as scratch between computes)
        ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
        if (t_mask) ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));

        // Update xt portion of input: [in_ch, T, N] (context_latents pre-filled)
        for (int b = 0; b < N; b++)
            for (int t = 0; t < T; t++)
                memcpy(&input_buf[b * T * in_ch + t * in_ch + ctx_ch],
                       &xt[b * n_per + t * Oc],
                       Oc * sizeof(float));
        ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));

        // compute forward pass (conditional)
        ggml_backend_sched_graph_compute(model->sched, gf);

        // dump intermediate tensors on step 0 (sample 0 only for batch)
        if (step == 0 && dbg && dbg->enabled) {
            auto dump_named = [&](const char *name) {
                struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                if (t) {
                    // For batched tensors, dump only sample 0 (first slice)
                    int64_t n0 = t->ne[0];
                    int64_t n1 = t->ne[1];
                    int64_t sample_elems = n0 * n1;  // [ne0, ne1] of first sample
                    std::vector<float> buf(sample_elems);
                    ggml_backend_tensor_get(t, buf.data(), 0, sample_elems * sizeof(float));
                    if (n1 <= 1) {
                        debug_dump_1d(dbg, name, buf.data(), (int)n0);
                    } else {
                        debug_dump_2d(dbg, name, buf.data(), (int)n0, (int)n1);
                    }
                }
            };
            dump_named("tproj");
            dump_named("temb");
            dump_named("temb_t");
            dump_named("temb_r");
            dump_named("sinusoidal_t");
            dump_named("sinusoidal_r");
            dump_named("temb_lin1_t");
            dump_named("temb_lin1_r");
            dump_named("hidden_after_proj_in");
            dump_named("proj_in_input");
            dump_named("enc_after_cond_emb");
            dump_named("layer0_sa_input");
            dump_named("layer0_q_after_rope");
            dump_named("layer0_k_after_rope");
            dump_named("layer0_sa_output");
            dump_named("layer0_attn_out");
            dump_named("layer0_after_self_attn");
            dump_named("layer0_after_cross_attn");
            dump_named("hidden_after_layer0");
            dump_named("hidden_after_layer6");
            dump_named("hidden_after_layer12");
            dump_named("hidden_after_layer18");
            dump_named("hidden_after_layer23");
        }

        // read velocity output: [Oc, T, N]
        ggml_backend_tensor_get(t_output, vt.data(), 0, n_total * sizeof(float));

        // CFG: unconditional pass + APG per sample
        if (do_cfg) {
            memcpy(vt_cond.data(), vt.data(), n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_cond", step);
                debug_dump_2d(dbg, name, vt_cond.data(), T, Oc);
            }

            // Unconditional pass: re-upload all inputs (scheduler clobbers input buffers during compute)
            ggml_backend_tensor_set(t_enc, null_enc_buf.data(), 0, H * enc_S * N * sizeof(float));
            ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));
            if (t_t) ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
            if (t_tr) ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
            ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
            if (t_mask) ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));

            ggml_backend_sched_graph_compute(model->sched, gf);
            ggml_backend_tensor_get(t_output, vt_uncond.data(), 0, n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_uncond", step);
                debug_dump_2d(dbg, name, vt_uncond.data(), T, Oc);
            }

            // APG per sample
            for (int b = 0; b < N; b++) {
                apg_forward(vt_cond.data() + b * n_per,
                            vt_uncond.data() + b * n_per,
                            guidance_scale, apg_mbufs[b],
                            vt.data() + b * n_per, Oc, T);
            }
        }

        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_2d(dbg, name, vt.data(), T, Oc);
        }

        // euler step (all N samples)
        if (step == num_steps - 1) {
            for (int i = 0; i < n_total; i++)
                output[i] = xt[i] - vt[i] * t_curr;
        } else {
            float dt = t_curr - schedule[step + 1];
            for (int i = 0; i < n_total; i++)
                xt[i] -= vt[i] * dt;
        }

        // debug dump (sample 0 only)
        if (dbg && dbg->enabled) {
            char name[64];
            if (step == num_steps - 1) {
                snprintf(name, sizeof(name), "dit_x0");
                debug_dump_2d(dbg, name, output, T, Oc);
            } else {
                snprintf(name, sizeof(name), "dit_step%d_xt", step);
                debug_dump_2d(dbg, name, xt.data(), T, Oc);
            }
        }

        fprintf(stderr, "[DiT] step %d/%d t=%.3f\n", step + 1, num_steps, t_curr);
    }

    // Batch diagnostic: report per-sample stats to catch corruption
    if (N >= 2) {
        for (int b = 0; b < N; b++) {
            const float * s = output + b * n_per;
            float mn = s[0], mx = s[0], sum = 0.0f;
            int n_nan = 0;
            for (int i = 0; i < n_per; i++) {
                float v = s[i];
                if (v != v) { n_nan++; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            fprintf(stderr, "[DiT] Batch%d output: min=%.4f max=%.4f mean=%.6f nan=%d\n",
                    b, mn, mx, sum / (float)n_per, n_nan);
        }
    }

    ggml_free(ctx);
}

// Free
static void dit_ggml_free(DiTGGML * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}

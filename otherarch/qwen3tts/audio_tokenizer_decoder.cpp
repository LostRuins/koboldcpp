#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

AudioTokenizerDecoder::AudioTokenizerDecoder() = default;

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    unload_model();
}

void AudioTokenizerDecoder::unload_model() {
    free_audio_decoder_model(model_);

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }

    state_.compute_meta.clear();
    codes_buf_.clear();
}

void AudioTokenizerDecoder::normalize_codebooks() {
    const float epsilon = 1e-5f;

    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char *) {
        if (!codebook || !usage || !codebook->data || !usage->data) return;

        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];

        ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data;
        float * usage_data = (float *)usage->data;

        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) u = epsilon;
            float inv_u = 1.0f / u;

            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }

    };

    normalize_codebook(model_.vq_first_codebook, model_.vq_first_usage, "first");

    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model_.vq_rest_codebook[i], model_.vq_rest_usage[i], name);
    }
}

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    model_.config.n_codebooks = loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16);
    model_.config.codebook_size = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);

    int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }

    if (dec_tensor_count == 0) {
        error_msg_ = "No decoder tensors found in model";
        return false;
    }

    size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
            continue;
        }



        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);

        model_.tensors[name] = tensor;

        std::string sname(name);

        if (sname == "tok_dec.vq_first.input_proj.weight") model_.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model_.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model_.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model_.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model_.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model_.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model_.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model_.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model_.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model_.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model_.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model_.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model_.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model_.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model_.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model_.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model_.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model_.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        }
        else {
            int blk_idx, res_idx, cb_idx, n = 0;
            char suffix[64];
            size_t name_len = strlen(name);



            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)

            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_codebook[cb_idx] = tensor;
                }
            }
            else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_usage[cb_idx] = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].conv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].dwconv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].norm_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv1_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv2_b = tensor;
                }
            }
            else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model_.upsample[blk_idx].gamma = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_q.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_k.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_v.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_output.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_gate.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_up.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_down.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_alpha = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_beta = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_w = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_b = tensor;
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_b = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_b = tensor;
                }
            }
            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_,
                                     qwen3tts_allowgpu)) {
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        model_.dec_blocks[i].res[0].dilation = 1;
        model_.dec_blocks[i].res[1].dilation = 3;
        model_.dec_blocks[i].res[2].dilation = 9;
    }

    normalize_codebooks();

    state_.backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg_, qwen3tts_allowgpu);
    if (!state_.backend) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerDecoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_DEC_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES + ggml_graph_overhead());

    return true;
}

struct ggml_tensor * AudioTokenizerDecoder::apply_snake(struct ggml_context * ctx,
                                                         struct ggml_tensor * x,
                                                         struct ggml_tensor * alpha,
                                                         struct ggml_tensor * beta) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    int64_t batch = x->ne[2];

    struct ggml_tensor * alpha_exp = ggml_exp(ctx, alpha);

    struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, alpha_exp, 1, channels, 1);
    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d,
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));

    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_broad);
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);

    struct ggml_tensor * neg_beta = ggml_scale(ctx, beta, -1.0f);
    struct ggml_tensor * inv_beta_exp = ggml_exp(ctx, neg_beta);
    struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, inv_beta_exp, 1, channels, 1);
    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d,
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));

    struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta);

    return ggml_add(ctx, x, scaled_sin);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_rms_norm(struct ggml_context * ctx,
                                                            struct ggml_tensor * x,
                                                            struct ggml_tensor * w,
                                                            float eps) {
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, w);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_pre_tfm_layer(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const pre_tfm_layer & layer,
                                                                 int32_t n_frames,
                                                                 struct ggml_tensor * positions) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int qkv_dim = cfg.latent_dim;
    const int head_dim = qkv_dim / n_heads;

    if (!layer.attn_norm_w || !layer.attn_q_w || !layer.attn_k_w || !layer.attn_v_w ||
        !layer.attn_output_w || !layer.ffn_norm_w || !layer.ffn_gate_w ||
        !layer.ffn_up_w || !layer.ffn_down_w) {
        return x;
    }

    struct ggml_tensor * residual = x;

    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, cfg.rms_norm_eps);

    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);

    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_frames);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_heads, n_frames);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_heads, n_frames);

    Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    struct ggml_tensor * K = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    struct ggml_tensor * V = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)head_dim));
    // Apply causal mask (each position can only attend to itself and previous positions)
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);

    V = ggml_cont(ctx, ggml_transpose(ctx, V));

    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);

    attn_out = ggml_mul_mat(ctx, layer.attn_output_w, attn_out);

    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }

    x = ggml_add(ctx, residual, attn_out);
    residual = x;

    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, cfg.rms_norm_eps);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate_w, normed);
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_w, normed);

    gate = ggml_silu(ctx, gate);
    struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, up);

    ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_out);

    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }

    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_upsample_block(struct ggml_context * ctx,
                                                                   struct ggml_tensor * x,
                                                                   const upsample_block & block,
                                                                   int block_idx) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];

     struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
     x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);

     int64_t new_seq_len = x_2d->ne[0];
     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);

     if (block.conv_b) {
         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
     }

     struct ggml_tensor * residual = x;

     if (block.dwconv_w) {
         // Causal padding: pad left with 6 zeros (kernel_size - 1 = 7 - 1 = 6)
         x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);  // left pad only
         x = ggml_conv_1d_dw(ctx, block.dwconv_w, x, 1, 0, 1);  // no padding in conv
         if (block.dwconv_b) {
             x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, 1, channels, 1));
         }
     }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

     if (block.norm_w && block.norm_b) {
         x = ggml_norm(ctx, x, 1e-6f);
         x = ggml_mul(ctx, x, block.norm_w);
         x = ggml_add(ctx, x, block.norm_b);
     }

     x = ggml_mul_mat(ctx, block.pwconv1_w, x);
     if (block.pwconv1_b) {
         x = ggml_add(ctx, x, block.pwconv1_b);
     }

     x = ggml_gelu(ctx, x);

     x = ggml_mul_mat(ctx, block.pwconv2_w, x);
     if (block.pwconv2_b) {
         x = ggml_add(ctx, x, block.pwconv2_b);
     }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

     if (block.gamma) {
         struct ggml_tensor * gamma_3d = ggml_reshape_3d(ctx, block.gamma, 1, channels, 1);
         x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d,
                                           ggml_new_tensor_3d(ctx, GGML_TYPE_F32, new_seq_len, channels, 1)));
     }

    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_residual_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const residual_block & block) {
    struct ggml_tensor * residual = x;

    if (block.act1_alpha) {
        x = apply_snake(ctx, x, block.act1_alpha, block.act1_beta);
    }

    int64_t out_channels = block.conv1_w->ne[2];
    int padding = 6 * block.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, block.conv1_w, x, 1, 0, block.dilation);
    if (block.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, 1, out_channels, 1));
    }

    if (block.act2_alpha) {
        x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta);
    }

    out_channels = block.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, block.conv2_w, x, 1, 0, 1);
    if (block.conv2_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv2_b, 1, out_channels, 1));
    }

    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_decoder_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const decoder_block & block,
                                                                  int upsample_rate,
                                                                  int block_idx) {
    if (block.snake_alpha && block.snake_beta) {
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta);
    }

     int64_t seq_len = x->ne[0];
     int64_t in_channels = x->ne[1];
     int64_t out_channels = block.conv_t_w->ne[1];
     int kernel_size = block.conv_t_w->ne[0];

     struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
     x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);

     int64_t new_seq_len = x_2d->ne[0];
     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);

     // Python CausalTransConvNet: left_pad = right_pad = kernel_size - stride
     int pad = kernel_size - upsample_rate;
     int left_pad = pad;
     int right_pad = pad;
     int64_t out_seq_len = new_seq_len - left_pad - right_pad;

     x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                      x->nb[1], x->nb[2], left_pad * x->nb[0]);
     x = ggml_cont(ctx, x);

     if (block.conv_t_b) {
         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
     }

    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
    }

    return x;
}

struct ggml_cgraph * AudioTokenizerDecoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_DEC_MAX_NODES, false);

    static const char * cb_names[16] = {
        "codes_cb0", "codes_cb1", "codes_cb2", "codes_cb3",
        "codes_cb4", "codes_cb5", "codes_cb6", "codes_cb7",
        "codes_cb8", "codes_cb9", "codes_cb10", "codes_cb11",
        "codes_cb12", "codes_cb13", "codes_cb14", "codes_cb15"
    };

    struct ggml_tensor * cb_codes_tensors[16];
    for (int cb = 0; cb < 16; ++cb) {
        cb_codes_tensors[cb] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(cb_codes_tensors[cb], cb_names[cb]);
        ggml_set_input(cb_codes_tensors[cb]);
    }

    struct ggml_tensor * first_codes = cb_codes_tensors[0];

     struct ggml_tensor * first_emb = ggml_get_rows(ctx0, model_.vq_first_codebook, first_codes);
     ggml_set_name(first_emb, "first_emb_raw");

     struct ggml_tensor * rest_emb[15];
     for (int cb = 0; cb < 15; ++cb) {
         struct ggml_tensor * cb_codes = cb_codes_tensors[cb + 1];
         rest_emb[cb] = ggml_get_rows(ctx0, model_.vq_rest_codebook[cb], cb_codes);

         if (cb == 0) {
             ggml_set_name(rest_emb[cb], "rest_cb0_emb_raw");
         }
     }

     struct ggml_tensor * first_emb_2d = ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim, n_frames);
     ggml_set_name(first_emb_2d, "first_emb_2d");

     struct ggml_tensor * first_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_first_output_proj,
                                                                   cfg.codebook_dim, cfg.hidden_dim);
     struct ggml_tensor * first_proj_2d = ggml_mul_mat(ctx0, first_proj_weight_2d, first_emb_2d);
     ggml_set_name(first_proj_2d, "first_proj_2d");

    struct ggml_tensor * rest_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_rest_output_proj,
                                                                 cfg.codebook_dim, cfg.hidden_dim);

     struct ggml_tensor * rest_proj_2d = nullptr;
     for (int cb = 0; cb < 15; ++cb) {
         struct ggml_tensor * cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);

         if (cb == 0) {
             ggml_set_name(cb_emb_2d, "rest_cb0_emb_2d");
         }

         struct ggml_tensor * cb_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, cb_emb_2d);

         if (rest_proj_2d == nullptr) {
             rest_proj_2d = cb_proj_2d;
         } else {
             rest_proj_2d = ggml_add(ctx0, rest_proj_2d, cb_proj_2d);
         }
     }
     ggml_set_name(rest_proj_2d, "rest_proj_2d");

     struct ggml_tensor * latent_2d = ggml_add(ctx0, first_proj_2d, rest_proj_2d);
     ggml_set_name(latent_2d, "latent_2d");

     struct ggml_tensor * latent_t = ggml_transpose(ctx0, latent_2d);
     ggml_set_name(latent_t, "latent_t");

     struct ggml_tensor * latent_cont = ggml_cont(ctx0, latent_t);
     ggml_set_name(latent_cont, "latent_cont");

     struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_cont, n_frames, cfg.hidden_dim, 1);

     ggml_set_name(latent, "vq_output");

    struct ggml_tensor * latent_for_conv = ggml_cont(ctx0, latent);
    struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0);
     struct ggml_tensor * cur = ggml_conv_1d(ctx0, model_.pre_conv_w, latent_padded, 1, 0, 1);
     if (model_.pre_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.pre_conv_b, 1, cfg.latent_dim, 1));
     }

     ggml_set_name(cur, "pre_conv_output");

     struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.latent_dim);
     struct ggml_tensor * cur_t = ggml_transpose(ctx0, cur_2d);
     cur = ggml_cont(ctx0, cur_t);

     ggml_set_name(cur, "pre_conv_reshaped");

     cur = ggml_mul_mat(ctx0, model_.pre_tfm_input_proj_w, cur);
     if (model_.pre_tfm_input_proj_b) {
         cur = ggml_add(ctx0, cur, model_.pre_tfm_input_proj_b);
     }

     ggml_set_name(cur, "pre_tfm_input");

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

     for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
         cur = apply_pre_tfm_layer(ctx0, cur, model_.pre_tfm_layers[i], n_frames, positions);
     }

     if (model_.pre_tfm_norm_w) {
         cur = apply_rms_norm(ctx0, cur, model_.pre_tfm_norm_w, cfg.rms_norm_eps);
     }

     cur = ggml_mul_mat(ctx0, model_.pre_tfm_output_proj_w, cur);
     if (model_.pre_tfm_output_proj_b) {
         cur = ggml_add(ctx0, cur, model_.pre_tfm_output_proj_b);
     }

     ggml_set_name(cur, "pre_tfm_output");

    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
     cur = ggml_cont(ctx0, cur);
     cur = ggml_reshape_3d(ctx0, cur, n_frames, cfg.latent_dim, 1);

     ggml_set_name(cur, "pre_tfm_reshaped");

     for (int i = 0; i < 2; ++i) {
         cur = apply_upsample_block(ctx0, cur, model_.upsample[i], i);
     }

     ggml_set_name(cur, "upsample_output");

     // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
     cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
     cur = ggml_conv_1d(ctx0, model_.dec0_conv_w, cur, 1, 0, 1);
     if (model_.dec0_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec0_conv_b, 1, cfg.decoder_dim, 1));
     }

     ggml_set_name(cur, "dec0_output");

     int upsample_rates[4] = {8, 5, 4, 3};
     for (int i = 0; i < 4; ++i) {
         cur = apply_decoder_block(ctx0, cur, model_.dec_blocks[i], upsample_rates[i], i);
         char name[32];
         snprintf(name, sizeof(name), "dec%d_output", i + 1);
         ggml_set_name(cur, name);
     }

     if (model_.dec5_snake_alpha) {
         cur = apply_snake(ctx0, cur, model_.dec5_snake_alpha, model_.dec5_snake_beta);
     }

     ggml_set_name(cur, "dec5_output");

     // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
     cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
     cur = ggml_conv_1d(ctx0, model_.dec6_conv_w, cur, 1, 0, 1);
     if (model_.dec6_conv_b) {
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec6_conv_b, 1, 1, 1));
     }

     ggml_set_name(cur, "dec6_output");

    cur = ggml_tanh(ctx0, cur);

    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);

    ggml_set_name(cur, "audio");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = model_.config;

    codes_buf_.resize(n_frames * cfg.n_codebooks);
    for (int f = 0; f < n_frames; ++f) {
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            codes_buf_[cb + f * cfg.n_codebooks] = codes[f * cfg.n_codebooks + cb];
        }
    }

    struct ggml_cgraph * gf = build_graph(n_frames);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }

    std::vector<int32_t> cb_codes(n_frames);
    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        struct ggml_tensor * cb_tensor = ggml_graph_get_tensor(gf, name);
        if (!cb_tensor) {
            error_msg_ = "Failed to find codes tensor for codebook " + std::to_string(cb);
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

        for (int f = 0; f < n_frames; ++f) {
            cb_codes[f] = codes_buf_[f * cfg.n_codebooks + cb];
        }

        ggml_backend_tensor_set(cb_tensor, cb_codes.data(), 0, n_frames * sizeof(int32_t));
    }



    struct ggml_tensor * positions_tensor = ggml_graph_get_tensor(gf, "positions");
    if (positions_tensor) {
        std::vector<int32_t> positions(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(positions_tensor, positions.data(), 0,
                                n_frames * sizeof(int32_t));
    }



    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    struct ggml_tensor * audio_tensor = ggml_graph_get_tensor(gf, "audio");
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));

    ggml_backend_sched_reset(state_.sched);

    return true;
}

void free_audio_decoder_model(audio_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts

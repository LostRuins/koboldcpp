#include "tts_transformer.h"
#include "gguf_loader.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <cstdlib>
#include <cctype>
#include <sys/stat.h>

#ifdef _WIN32
#    define ttsfseek _fseeki64
#else
#    define ttsfseek fseeko
#endif

bool qwen3tts_allowgpu = false;

namespace qwen3_tts {

TTSTransformer::TTSTransformer() = default;

struct ggml_tensor * TTSTransformer::mul_mat(struct ggml_context * ctx,
                                              struct ggml_tensor * a,
                                              struct ggml_tensor * b) {
    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    if (force_f32_acc_) {
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    return result;
}

TTSTransformer::~TTSTransformer() {
    unload_model();
}

void TTSTransformer::unload_model() {
    free_tts_kv_cache(state_.cache);
    free_tts_kv_cache(state_.code_pred_cache);
    free_transformer_model(model_);

    coreml_code_predictor_.unload();
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();
    skip_ggml_code_pred_layers_ = false;

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
    last_hidden_.clear();
    embd_row_fp16_scratch_.clear();
}

void TTSTransformer::set_seed(int seed)
{
    if (seed <= 0 || seed==0xFFFFFFFF)
    {
        seed = (((uint32_t)time(NULL)) % 1000000u);
    }
    this->rng_ = std::mt19937(seed);
}

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    skip_ggml_code_pred_layers_ = false;

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }

    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!load_tensor_data(model_path, ctx)) {
        free_transformer_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);

    state_.backend = init_preferred_backend("TTSTransformer", &error_msg_, qwen3tts_allowgpu);
    if (!state_.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());

    if (!try_init_coreml_code_predictor(model_path)) {
        return false;
    }

    return true;
}

bool TTSTransformer::try_init_coreml_code_predictor(const std::string & model_path) {
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();

    return true;
}

bool TTSTransformer::parse_config(struct gguf_context * ctx) {
    auto get_u32_any = [&](std::initializer_list<const char *> keys, int32_t default_val) -> int32_t {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return (int32_t)gguf_get_val_u32(ctx, idx);
            }
        }
        return default_val;
    };

    auto get_f32_any = [&](std::initializer_list<const char *> keys, float default_val) -> float {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return gguf_get_val_f32(ctx, idx);
            }
        }
        return default_val;
    };

    auto & cfg = model_.config;
    cfg.text_vocab_size = get_u32_any({
        "qwen3-tts.text.vocab_size",
        "qwen3-tts.text_vocab_size",
    }, 151936);
    cfg.text_embd_dim = get_u32_any({
        "qwen3-tts.text.embedding_dim",
        "qwen3-tts.text_hidden_size",
    }, 2048);
    cfg.hidden_size = get_u32_any({
        "qwen3-tts.talker.embedding_length",
        "qwen3-tts.embedding_length",
    }, 1024);
    cfg.n_layers = get_u32_any({
        "qwen3-tts.talker.block_count",
        "qwen3-tts.block_count",
    }, 28);
    cfg.n_attention_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count",
        "qwen3-tts.attention.head_count",
    }, 16);
    cfg.n_key_value_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count_kv",
        "qwen3-tts.attention.head_count_kv",
    }, 8);
    cfg.intermediate_size = get_u32_any({
        "qwen3-tts.talker.feed_forward_length",
        "qwen3-tts.feed_forward_length",
    }, 3072);
    cfg.head_dim = get_u32_any({
        "qwen3-tts.talker.attention.key_length",
        "qwen3-tts.attention.key_length",
    }, 128);
    cfg.rms_norm_eps = get_f32_any({
        "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
        "qwen3-tts.attention.layer_norm_rms_epsilon",
    }, 1e-6f);
    cfg.rope_theta = get_f32_any({
        "qwen3-tts.talker.rope.freq_base",
        "qwen3-tts.rope.freq_base",
    }, 1000000.0f);

    cfg.codec_vocab_size = get_u32_any({
        "qwen3-tts.talker.codec_vocab_size",
        "qwen3-tts.vocab_size",
    }, 3072);
    cfg.n_codebooks = get_u32_any({
        "qwen3-tts.talker.num_codebooks",
        "qwen3-tts.num_code_groups",
    }, 16);

    cfg.code_pred_layers = get_u32_any({
        "qwen3-tts.code_pred.layer_count",
        "qwen3-tts.code_predictor.layer_count",
    }, 5);
    cfg.code_pred_vocab_size = get_u32_any({
        "qwen3-tts.code_pred.vocab_size",
        "qwen3-tts.code_predictor.vocab_size",
    }, 2048);

    cfg.code_pred_hidden_size = get_u32_any({
        "qwen3-tts.code_pred.embedding_length",
    }, cfg.hidden_size);  // default to talker hidden_size (0.6B case)
    cfg.code_pred_n_attention_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count",
    }, cfg.n_attention_heads);
    cfg.code_pred_n_key_value_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count_kv",
    }, cfg.n_key_value_heads);
    cfg.code_pred_intermediate_size = get_u32_any({
        "qwen3-tts.code_pred.feed_forward_length",
    }, cfg.intermediate_size);
    cfg.code_pred_head_dim = get_u32_any({
        "qwen3-tts.code_pred.attention.key_length",
    }, cfg.head_dim);

    cfg.codec_pad_id = get_u32_any({
        "qwen3-tts.codec.pad_id",
    }, 2148);
    cfg.codec_bos_id = get_u32_any({
        "qwen3-tts.codec.bos_id",
    }, 2149);
    cfg.codec_eos_id = get_u32_any({
        "qwen3-tts.codec.eos_id",
        "qwen3-tts.codec.eos_token_id",
    }, 2150);

    cfg.tts_bos_token_id = get_u32_any({
        "qwen3-tts.tts_bos_token_id",
        "qwen3-tts.tts.bos_token_id",
        "qwen3-tts.tts.bos_id",
    }, 151672);
    cfg.tts_eos_token_id = get_u32_any({
        "qwen3-tts.tts_eos_token_id",
        "qwen3-tts.tts.eos_token_id",
        "qwen3-tts.tts.eos_id",
    }, 151673);
    cfg.tts_pad_token_id = get_u32_any({
        "qwen3-tts.tts_pad_token_id",
        "qwen3-tts.tts.pad_token_id",
        "qwen3-tts.tts.pad_id",
    }, 151671);

    cfg.codec_think_id = get_u32_any({
        "qwen3-tts.codec.think_id",
        "qwen3-tts.codec_think_id",
    }, 2154);
    cfg.codec_nothink_id = get_u32_any({
        "qwen3-tts.codec.nothink_id",
        "qwen3-tts.codec_nothink_id",
    }, 2155);
    cfg.codec_think_bos_id = get_u32_any({
        "qwen3-tts.codec.think_bos_id",
        "qwen3-tts.codec_think_bos_id",
    }, 2156);
    cfg.codec_think_eos_id = get_u32_any({
        "qwen3-tts.codec.think_eos_id",
        "qwen3-tts.codec_think_eos_id",
    }, 2157);

    cfg.english_language_id = get_u32_any({
        "qwen3-tts.language.english_id",
        "qwen3-tts.codec.language.english_id",
        "qwen3-tts.language_id",
    }, 2050);

    return true;
}

bool TTSTransformer::create_tensors(struct gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;

    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }

    model_.layers.resize(cfg.n_layers);
    model_.code_pred_layers.resize(cfg.code_pred_layers);
    model_.code_pred_embd.resize(cfg.n_codebooks - 1);
    model_.code_pred_head.resize(cfg.n_codebooks - 1);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;

        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }

        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.n_layers) {

                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            if (skip_ggml_code_pred_layers_) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {

                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.code_pred_intermediate_size;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.mtp_proj.")) {
            if (strstr(name, "weight")) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_hidden_size;
                n_dims = 2;
            } else if (strstr(name, "bias")) {
                ne[0] = cfg.code_pred_hidden_size;
                n_dims = 1;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
         } else if (strstr(name, "code_pred.lm_head.")) {
             if (skip_ggml_code_pred_layers_) {
                 continue;
             }
             int cb_idx = -1;
             if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                 cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                 ne[0] = cfg.code_pred_hidden_size;
                 ne[1] = cfg.code_pred_vocab_size;
                 n_dims = 2;
             } else {
                 continue;
             }
         } else if (strstr(name, "code_pred.output_norm.weight")) {
             if (skip_ggml_code_pred_layers_) {
                 continue;
             }
             ne[0] = cfg.code_pred_hidden_size;
             n_dims = 1;
         } else {
             continue;
         }

        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;

        if (strstr(name, "talker.text_embd.weight")) {
            model_.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            model_.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            model_.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            model_.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            model_.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            model_.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            model_.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = model_.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = model_.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                model_.code_pred_embd[cb_idx] = tensor;
            }
         } else if (strstr(name, "code_pred.lm_head.")) {
             int cb_idx = -1;
             sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
             if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                 model_.code_pred_head[cb_idx] = tensor;
             }
         } else if (strstr(name, "code_pred.output_norm.weight")) {
             model_.code_pred_output_norm = tensor;
         } else if (strstr(name, "code_pred.mtp_proj.weight")) {
             model_.code_pred_mtp_proj_w = tensor;
         } else if (strstr(name, "code_pred.mtp_proj.bias")) {
             model_.code_pred_mtp_proj_b = tensor;
         }
     }

     return true;
 }

bool TTSTransformer::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
    ggml_backend_t backend = init_preferred_backend("TTSTransformer", &error_msg_, qwen3tts_allowgpu);
    if (!backend) {
        return false;
    }

    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, backend);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate tensor buffer";
        release_preferred_backend(backend);
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg_ = "Failed to open file for reading: " + path;
        release_preferred_backend(backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = model_.tensors.find(name);
        if (it == model_.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

        if (ttsfseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg_ = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg_ = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    release_preferred_backend(backend);

    return true;
}

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;

    free_tts_kv_cache(state_.cache);

    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_layers;

    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }

    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);

    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);

        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }

    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_kv_cache() {
    state_.cache.n_used = 0;
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;

    free_tts_kv_cache(state_.code_pred_cache);

    state_.code_pred_cache.n_ctx = n_ctx;
    state_.code_pred_cache.n_used = 0;
    state_.code_pred_cache.head_dim = cfg.code_pred_head_dim;
    state_.code_pred_cache.n_kv_heads = cfg.code_pred_n_key_value_heads;
    state_.code_pred_cache.n_layers = cfg.code_pred_layers;

    const size_t n_tensors = cfg.code_pred_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    state_.code_pred_cache.ctx = ggml_init(params);
    if (!state_.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }

    state_.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    state_.code_pred_cache.v_cache.resize(cfg.code_pred_layers);

    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        state_.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);

        state_.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }

    state_.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(state_.code_pred_cache.ctx, state_.backend);
    if (!state_.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    state_.code_pred_cache.n_used = 0;
}

bool TTSTransformer::lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                                           int32_t n_tokens, const char * input_name,
                                           const char * output_name, std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    const int32_t embd_dim = (int32_t) embedding->ne[0];
    if (n_tokens <= 32 &&
        (embedding->type == GGML_TYPE_F16 || embedding->type == GGML_TYPE_F32)) {
        output.resize((size_t) embd_dim * n_tokens);
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (!lookup_single_embedding_row(embedding, token_ids[t],
                                             output.data() + (size_t) t * embd_dim)) {
                return false;
            }
        }
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, input_name);
    ggml_set_input(inp_tokens);

    struct ggml_tensor * rows = ggml_get_rows(ctx0, embedding, inp_tokens);
    rows = ggml_cast(ctx0, rows, GGML_TYPE_F32);
    ggml_set_name(rows, output_name);
    ggml_set_output(rows);

    ggml_build_forward_expand(gf, rows);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate embedding lookup graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, input_name);
    if (!inp) {
        error_msg_ = std::string("Failed to find input tensor: ") + input_name;
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp, token_ids, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute embedding lookup graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, output_name);
    if (!out) {
        error_msg_ = "Failed to find embedding lookup output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)embedding->ne[0] * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

bool TTSTransformer::lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                                 float * out_row) {
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (!out_row) {
        error_msg_ = "Embedding output row is null";
        return false;
    }

    const int64_t embd_dim = embedding->ne[0];
    const int64_t vocab_size = embedding->ne[1];
    if (token_id < 0 || token_id >= vocab_size) {
        error_msg_ = "Embedding token ID out of range";
        return false;
    }

    const size_t row_offset = (size_t) token_id * embedding->nb[1];
    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embedding, out_row, row_offset, (size_t) embd_dim * sizeof(float));
        return true;
    }
    if (embedding->type == GGML_TYPE_F16) {
        embd_row_fp16_scratch_.resize((size_t) embd_dim);
        ggml_backend_tensor_get(embedding, embd_row_fp16_scratch_.data(),
                                row_offset, (size_t) embd_dim * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < embd_dim; ++i) {
            out_row[i] = ggml_fp16_to_fp32(embd_row_fp16_scratch_[i]);
        }
        return true;
    }

    std::vector<int32_t> single_token = { token_id };
    std::vector<float> single_out;
    if (!lookup_embedding_rows(embedding, single_token.data(), 1,
                               "inp_compat_embed", "out_compat_embed", single_out)) {
        return false;
    }
    memcpy(out_row, single_out.data(), (size_t) embd_dim * sizeof(float));
    return true;
}

bool TTSTransformer::project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                                         std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_text_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.text_embd, inp_tokens);
    cur = mul_mat(ctx0, model_.text_proj_fc1, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc1_bias);
    cur = ggml_silu(ctx0, cur);
    cur = mul_mat(ctx0, model_.text_proj_fc2, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc2_bias);

    ggml_set_name(cur, "text_proj_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate text projection graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp_text_tokens");
    if (!inp) {
        error_msg_ = "Failed to find inp_text_tokens tensor in graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp, text_tokens, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute text projection graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "text_proj_out");
    if (!out) {
        error_msg_ = "Failed to find text projection output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)model_.config.hidden_size * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

bool TTSTransformer::build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                                         const float * speaker_embd, int32_t language_id,
                                         std::vector<float> & prefill_embd,
                                         std::vector<float> & trailing_text_hidden,
                                         std::vector<float> & tts_pad_embed,
                                         int32_t speaker_token_id,
                                         const int32_t * instruct_tokens,
                                         int32_t n_instruct_tokens) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for prefill";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t hidden_size = cfg.hidden_size;

    int32_t special_tokens[3] = {
        cfg.tts_bos_token_id,
        cfg.tts_eos_token_id,
        cfg.tts_pad_token_id,
    };

    std::vector<float> special_proj;
    if (!project_text_tokens(special_tokens, 3, special_proj)) {
        return false;
    }

    std::vector<float> tts_bos_embed(hidden_size);
    std::vector<float> tts_eos_embed(hidden_size);
    tts_pad_embed.resize(hidden_size);
    memcpy(tts_bos_embed.data(), special_proj.data() + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_eos_embed.data(), special_proj.data() + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_pad_embed.data(), special_proj.data() + 2 * hidden_size, hidden_size * sizeof(float));

    std::vector<float> role_embed;
    if (!project_text_tokens(text_tokens, 3, role_embed)) {
        return false;
    }

    std::vector<int32_t> codec_prefill_tokens;
    if (language_id < 0) {
        codec_prefill_tokens = {
            cfg.codec_nothink_id,
            cfg.codec_think_bos_id,
            cfg.codec_think_eos_id,
        };
    } else {
        codec_prefill_tokens = {
            cfg.codec_think_id,
            cfg.codec_think_bos_id,
            language_id,
            cfg.codec_think_eos_id,
        };
    }

    std::vector<float> codec_prefill_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_prefill_tokens.data(),
                               (int32_t)codec_prefill_tokens.size(),
                               "inp_codec_prefill_tokens", "codec_prefill_rows",
                               codec_prefill_embed)) {
        return false;
    }

    int32_t codec_tail_tokens[2] = { cfg.codec_pad_id, cfg.codec_bos_id };
    std::vector<float> codec_tail_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_tail_tokens, 2,
                               "inp_codec_tail_tokens", "codec_tail_rows",
                               codec_tail_embed)) {
        return false;
    }

    // If speaker_token_id is set, look up codec_embd[speaker_token_id] as the speaker embedding
    std::vector<float> speaker_token_embed;
    const float * effective_speaker_embd = speaker_embd;
    if (speaker_token_id >= 0) {
        speaker_token_embed.resize(hidden_size);
        if (!lookup_single_embedding_row(model_.codec_embd, speaker_token_id, speaker_token_embed.data())) {
            return false;
        }
        effective_speaker_embd = speaker_token_embed.data();
    }

    const bool has_speaker = (effective_speaker_embd != nullptr);
    const int32_t codec_input_len = (int32_t)codec_prefill_tokens.size() + (has_speaker ? 1 : 0) + 2;
    std::vector<float> codec_input_embedding((size_t)codec_input_len * hidden_size);

    int32_t dst_token = 0;
    memcpy(codec_input_embedding.data(), codec_prefill_embed.data(), codec_prefill_embed.size() * sizeof(float));
    dst_token += (int32_t)codec_prefill_tokens.size();

    if (has_speaker) {
        memcpy(codec_input_embedding.data() + (size_t)dst_token * hidden_size,
               effective_speaker_embd, hidden_size * sizeof(float));
        ++dst_token;
    }

    memcpy(codec_input_embedding.data() + (size_t)dst_token * hidden_size,
           codec_tail_embed.data(), codec_tail_embed.size() * sizeof(float));

    const int32_t codec_plus_overlay_len = codec_input_len - 1;
    std::vector<float> codec_plus_overlay((size_t)codec_plus_overlay_len * hidden_size);
    for (int32_t t = 0; t < codec_plus_overlay_len; ++t) {
        const float * overlay = (t == codec_plus_overlay_len - 1)
            ? tts_bos_embed.data()
            : tts_pad_embed.data();
        const float * codec_row = codec_input_embedding.data() + (size_t)t * hidden_size;
        float * out_row = codec_plus_overlay.data() + (size_t)t * hidden_size;
        for (int32_t h = 0; h < hidden_size; ++h) {
            out_row[h] = overlay[h] + codec_row[h];
        }
    }

    std::vector<float> first_text_embed;
    if (!project_text_tokens(text_tokens + 3, 1, first_text_embed)) {
        return false;
    }

    std::vector<float> first_text_plus_codec_bos(hidden_size);
    const float * codec_bos_embed = codec_input_embedding.data() + (size_t)(codec_input_len - 1) * hidden_size;
    for (int32_t h = 0; h < hidden_size; ++h) {
        first_text_plus_codec_bos[h] = first_text_embed[h] + codec_bos_embed[h];
    }

    // Project instruct tokens if provided
    std::vector<float> instruct_proj;
    if (instruct_tokens && n_instruct_tokens > 0) {
        if (!project_text_tokens(instruct_tokens, n_instruct_tokens, instruct_proj)) {
            return false;
        }
    }
    const int32_t instruct_len = (int32_t)(instruct_proj.size() / hidden_size);

    const int32_t prefill_len = instruct_len + 3 + codec_plus_overlay_len + 1;
    prefill_embd.resize((size_t)prefill_len * hidden_size);
    int32_t pos = 0;
    // instruct projection first (matches Python ordering: instruct before role)
    if (instruct_len > 0) {
        memcpy(prefill_embd.data(), instruct_proj.data(), (size_t)instruct_len * hidden_size * sizeof(float));
        pos += instruct_len;
    }
    // role_embed (3 tokens)
    memcpy(prefill_embd.data() + (size_t)pos * hidden_size,
           role_embed.data(), (size_t)3 * hidden_size * sizeof(float));
    pos += 3;
    // codec_plus_overlay
    memcpy(prefill_embd.data() + (size_t)pos * hidden_size,
           codec_plus_overlay.data(), (size_t)codec_plus_overlay_len * hidden_size * sizeof(float));
    pos += codec_plus_overlay_len;
    // first_text + codec_bos (last token)
    memcpy(prefill_embd.data() + (size_t)pos * hidden_size,
           first_text_plus_codec_bos.data(), hidden_size * sizeof(float));

    const int32_t trailing_token_count = std::max(0, n_tokens - 9);
    std::vector<float> trailing_text_proj;
    if (trailing_token_count > 0) {
        if (!project_text_tokens(text_tokens + 4, trailing_token_count, trailing_text_proj)) {
            return false;
        }
    }

    const int32_t trailing_len = trailing_token_count + 1;
    trailing_text_hidden.resize((size_t)trailing_len * hidden_size);
    if (trailing_token_count > 0) {
        memcpy(trailing_text_hidden.data(), trailing_text_proj.data(), trailing_text_proj.size() * sizeof(float));
    }
    memcpy(trailing_text_hidden.data() + (size_t)(trailing_len - 1) * hidden_size,
           tts_eos_embed.data(), hidden_size * sizeof(float));

    return true;
}

struct ggml_cgraph * TTSTransformer::build_prefill_forward_graph(int32_t n_tokens, int32_t n_past) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_prefill_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(inp_prefill_embd, "inp_prefill_embd");
    ggml_set_input(inp_prefill_embd);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur = inp_prefill_embd;

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf(float(head_dim));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];

        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);

        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

        int n_kv = n_past + n_tokens;

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        struct ggml_tensor * KQ = mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

        struct ggml_tensor * KQV = mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);

        cur = mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);

        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = mul_mat(ctx0, ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph * TTSTransformer::build_step_graph(int32_t n_past) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_step_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, 1);
    ggml_set_name(inp_step_embd, "inp_step_embd");
    ggml_set_input(inp_step_embd);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur = inp_step_embd;

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf(float(head_dim));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];

        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);

        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

        int n_kv = n_past + n_tokens;

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        struct ggml_tensor * KQ = mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

        struct ggml_tensor * KQV = mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);

        cur = mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);

        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = mul_mat(ctx0, ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_graph(int32_t n_prev_codes) {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int cp_hidden = cfg.code_pred_hidden_size;
    const int talker_hidden = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const int n_layer = cfg.code_pred_layers;
    const int n_codebooks = cfg.n_codebooks;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hidden);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    struct ggml_tensor * inp_prev_codes = nullptr;
    if (n_prev_codes > 0) {
        inp_prev_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prev_codes);
        ggml_set_name(inp_prev_codes, "inp_prev_codes");
        ggml_set_input(inp_prev_codes);
    }

    // Project talker hidden to code_pred hidden
    struct ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_hidden, talker_hidden, 1);
    if (model_.code_pred_mtp_proj_w) {
        cur = mul_mat(ctx0, model_.code_pred_mtp_proj_w, cur);
        if (model_.code_pred_mtp_proj_b) {
            cur = ggml_add(ctx0, cur, model_.code_pred_mtp_proj_b);
        }
    }

    if (n_prev_codes > 0 && inp_prev_codes) {
        for (int cb = 0; cb < n_prev_codes && cb < n_codebooks - 1; ++cb) {
            struct ggml_tensor * code_idx = ggml_view_1d(ctx0, inp_prev_codes, 1, cb * sizeof(int32_t));
            struct ggml_tensor * code_embd_raw = ggml_get_rows(ctx0, model_.code_pred_embd[cb], code_idx);
            // code_pred_embd is [talker_hidden, vocab] — project down
            struct ggml_tensor * code_embd = code_embd_raw;
            if (model_.code_pred_mtp_proj_w) {
                code_embd = mul_mat(ctx0, model_.code_pred_mtp_proj_w, code_embd_raw);
                if (model_.code_pred_mtp_proj_b) {
                    code_embd = ggml_add(ctx0, code_embd, model_.code_pred_mtp_proj_b);
                }
            }
            cur = ggml_add(ctx0, cur, code_embd);
        }
    }

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf(float(head_dim));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, 1);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

        struct ggml_tensor * KQ = mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

        struct ggml_tensor * KQV = mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, 1);

        cur = mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);

        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * old_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = mul_mat(ctx0, old_ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // Apply output normalization (matching prefill/step paths)
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);

    std::vector<struct ggml_tensor *> all_logits;
    for (int cb = 0; cb < n_codebooks - 1; ++cb) {
        struct ggml_tensor * cb_logits = mul_mat(ctx0, model_.code_pred_head[cb], cur);
        ggml_format_name(cb_logits, "logits_cb%d", cb + 1);
        ggml_set_output(cb_logits);
        all_logits.push_back(cb_logits);
    }

    for (auto * logits : all_logits) {
        ggml_build_forward_expand(gf, logits);
    }

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_prefill_graph() {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int cp_hidden = cfg.code_pred_hidden_size;
    const int talker_hidden = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 2;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    // Input: past_hidden from talker [talker_hidden]
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hidden);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    // Input: codebook 0 token embedding [talker_hidden] (pre-computed using talker's codec_embd)
    struct ggml_tensor * inp_cb0_embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hidden);
    ggml_set_name(inp_cb0_embd, "inp_cb0_embd");
    ggml_set_input(inp_cb0_embd);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // Concatenate [past_hidden, cb0_embd] -> [talker_hidden, 2]
    struct ggml_tensor * hidden_2d = ggml_reshape_2d(ctx0, inp_hidden, talker_hidden, 1);
    struct ggml_tensor * cb0_2d = ggml_reshape_2d(ctx0, inp_cb0_embd, talker_hidden, 1);
    struct ggml_tensor * cur = ggml_concat(ctx0, hidden_2d, cb0_2d, 1);

    // Apply MTP projection if present (talker_hidden -> code_pred_hidden)
    if (model_.code_pred_mtp_proj_w) {
        cur = mul_mat(ctx0, model_.code_pred_mtp_proj_w, cur);
        if (model_.code_pred_mtp_proj_b) {
            cur = ggml_add(ctx0, cur, model_.code_pred_mtp_proj_b);
        }
    }

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf(float(head_dim));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];

        // Store at position 0 (prefill starts fresh)
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2], 0);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

        struct ggml_tensor * KQ = mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, 0);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

        struct ggml_tensor * KQV = mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);

        cur = mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);

        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = mul_mat(ctx0, ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

     cur = inpL;

     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);

     struct ggml_tensor * last_hidden = ggml_view_2d(ctx0, cur, cp_hidden, 1,
                                                      cur->nb[1], cp_hidden * sizeof(float));

     struct ggml_tensor * logits = mul_mat(ctx0, model_.code_pred_head[0], last_hidden);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_step_graph(int32_t n_past, int32_t generation_step) {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int cp_hidden = cfg.code_pred_hidden_size;
    const int talker_hidden = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    // inp_hidden is only used when generation_step == 0 (not used in step graph normally)
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hidden);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    struct ggml_tensor * inp_code = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code, "inp_code");
    ggml_set_input(inp_code);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur;
    if (generation_step == 0) {
        // inp_hidden is talker-dim, project down
        cur = ggml_reshape_2d(ctx0, inp_hidden, talker_hidden, 1);
        if (model_.code_pred_mtp_proj_w) {
            cur = mul_mat(ctx0, model_.code_pred_mtp_proj_w, cur);
            if (model_.code_pred_mtp_proj_b) {
                cur = ggml_add(ctx0, cur, model_.code_pred_mtp_proj_b);
            }
        }
    } else {
        // code_pred_embd is [talker_hidden, vocab] — lookup returns talker-dim, project down
        cur = ggml_get_rows(ctx0, model_.code_pred_embd[generation_step - 1], inp_code);
        cur = ggml_reshape_2d(ctx0, cur, talker_hidden, 1);
        if (model_.code_pred_mtp_proj_w) {
            cur = mul_mat(ctx0, model_.code_pred_mtp_proj_w, cur);
            if (model_.code_pred_mtp_proj_b) {
                cur = ggml_add(ctx0, cur, model_.code_pred_mtp_proj_b);
            }
        }
    }

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf(float(head_dim));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];

        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);

        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

        int n_kv = n_past + n_tokens;

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        struct ggml_tensor * KQ = mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

        struct ggml_tensor * KQV = mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);

        cur = mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);

        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * step_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = mul_mat(ctx0, step_ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

     cur = inpL;

     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);

     struct ggml_tensor * logits = mul_mat(ctx0, model_.code_pred_head[generation_step], cur);
     ggml_set_name(logits, "logits");
     ggml_set_output(logits);

     ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);

    return gf;
}

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_prefill_forward_graph(n_tokens, n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t)n_tokens * model_.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    output.resize(n_tokens * model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));

    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(),
                           (n_tokens - 1) * model_.config.hidden_size * sizeof(float),
                           model_.config.hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

        logits_out->resize(model_.config.codec_vocab_size);
        ggml_backend_tensor_get(logits, logits_out->data(),
                                (n_tokens - 1) * model_.config.codec_vocab_size * sizeof(float),
                                model_.config.codec_vocab_size * sizeof(float));
    }

    state_.cache.n_used = n_past + n_tokens;

    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    return true;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t n_past,
                                  std::vector<float> & output) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    std::vector<float> projected;
    if (!project_text_tokens(text_tokens, n_tokens, projected)) {
        return false;
    }

    if (speaker_embd) {
        const int32_t hidden_size = model_.config.hidden_size;
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * row = projected.data() + (size_t)t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) {
                row[h] += speaker_embd[h];
            }
        }
    }

    return forward_prefill(projected.data(), n_tokens, n_past, output, nullptr);
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!step_embd) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_step_graph(n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
    if (inp_step) {
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                model_.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor in step graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(), 0,
                           model_.config.hidden_size * sizeof(float));
    if (hidden_out) {
        *hidden_out = last_hidden_;
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    output.resize(model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));

    state_.cache.n_used = n_past + 1;

    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    return true;
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                   std::vector<float> & output) {
    std::vector<float> codec_row;
    if (!lookup_embedding_rows(model_.codec_embd, &codec_token, 1,
                               "inp_legacy_codec_token", "legacy_codec_row",
                               codec_row)) {
        return false;
    }

    return forward_step(codec_row.data(), n_past, output, nullptr);
}

bool TTSTransformer::get_hidden_states(std::vector<float> & hidden) const {
    if (last_hidden_.empty()) {
        return false;
    }
    hidden = last_hidden_;
    return true;
}

bool TTSTransformer::predict_codes(const float * hidden, const int32_t * prev_codes,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = model_.config;
    int n_prev = (prev_codes != nullptr) ? cfg.n_codebooks - 1 : 0;

    struct ggml_cgraph * gf = build_code_pred_graph(n_prev);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate code predictor graph";
        return false;
    }

    struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
    if (inp_hidden) {
        ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
    }

    if (n_prev > 0) {
        struct ggml_tensor * inp_prev = ggml_graph_get_tensor(gf, "inp_prev_codes");
        if (inp_prev) {
            ggml_backend_tensor_set(inp_prev, prev_codes, 0, n_prev * sizeof(int32_t));
        }
    }

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute code predictor graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    output.resize((cfg.n_codebooks - 1) * cfg.code_pred_vocab_size);

    for (int cb = 0; cb < cfg.n_codebooks - 1; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "logits_cb%d", cb + 1);
        struct ggml_tensor * cb_logits = ggml_graph_get_tensor(gf, name);
        if (cb_logits) {
            ggml_backend_tensor_get(cb_logits, output.data() + cb * cfg.code_pred_vocab_size,
                                   0, cfg.code_pred_vocab_size * sizeof(float));
        }
    }

    ggml_backend_sched_reset(state_.sched);

    return true;
}

static int32_t argmax(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bool TTSTransformer::predict_codes_autoregressive_coreml(const float * hidden,
                                                         int32_t codebook_0_token,
                                                         std::vector<int32_t> & output,
                                                         float temperature,
                                                         int32_t top_k) {
    if (!use_coreml_code_predictor_ || !coreml_code_predictor_.is_loaded()) {
        error_msg_ = "CoreML code predictor is not loaded";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t n_steps = cfg.n_codebooks - 1;

    output.resize(n_steps);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t)cfg.n_codebooks * cfg.hidden_size, 0.0f);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }

        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }

        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }

        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }

        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };

    memcpy(seq_embd.data(), hidden, (size_t)cfg.hidden_size * sizeof(float));
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token,
                                     seq_embd.data() + cfg.hidden_size)) {
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    for (int32_t step = 0; step < n_steps; ++step) {
        if (step > 0) {
            float * dst = seq_embd.data() + (size_t)(step + 1) * cfg.hidden_size;
            if (!lookup_single_embedding_row(model_.code_pred_embd[step - 1], output[step - 1], dst)) {
                return false;
            }
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!coreml_code_predictor_.predict_step(step, seq_embd.data(), step + 2, cfg.hidden_size, logits_data)) {
            error_msg_ = "CoreML predictor step failed: " + coreml_code_predictor_.get_error();
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_compute_ms += dt_ms;
        if (timing_) timing_->t_code_pred_coreml_ms += dt_ms;
#endif

        if ((int32_t)logits_data.size() != cfg.code_pred_vocab_size) {
            error_msg_ = "CoreML predictor returned unexpected logits size";
            return false;
        }
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

#ifdef QWEN3_TTS_TIMING
        if (timing_) {
            if (step == 0) {
                timing_->t_code_pred_prefill_ms += dt_ms;
            } else {
                timing_->t_code_pred_steps_ms += dt_ms;
            }
        }
#endif
    }

    return true;
}

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                   std::vector<int32_t> & output,
                                                   float temperature, int32_t top_k) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = model_.config;

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    if (use_coreml_code_predictor_ && coreml_code_predictor_.is_loaded()) {
        if (predict_codes_autoregressive_coreml(hidden, codebook_0_token, output, temperature, top_k)) {
            return true;
        }
        if (skip_ggml_code_pred_layers_) {
            return false;
        }
        fprintf(stderr, "  CoreML code predictor failed, falling back to GGML: %s\n", error_msg_.c_str());
        use_coreml_code_predictor_ = false;
    }

    if (!model_.code_pred_output_norm) {
        error_msg_ = "code_pred_output_norm not loaded (required for GGML code predictor)";
        return false;
    }
    if (model_.code_pred_head.empty()) {
        error_msg_ = "code_pred_head tensors not loaded (required for GGML code predictor)";
        return false;
    }

    if (state_.code_pred_cache.n_ctx < cfg.n_codebooks) {
        if (!init_code_pred_kv_cache(cfg.n_codebooks)) {
            return false;
        }
    }
    clear_code_pred_kv_cache();

    output.resize(cfg.n_codebooks - 1);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);

    std::vector<float> code_probs(cfg.code_pred_vocab_size);

    // Helper lambda: temperature + top-k sampling (or greedy if temperature <= 0)
    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }
        // Temperature scaling
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }
        // Top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }
        // Softmax
        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }
        // Sample
        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };

    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // Prefill with 2 tokens [past_hidden, cb0_embd]
    {
#ifdef QWEN3_TTS_TIMING
        auto t_pf_start = clk::now();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_prefill_graph();
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor prefill graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_cb0_embd = ggml_graph_get_tensor(gf, "inp_cb0_embd");
        if (inp_cb0_embd) {
            ggml_backend_tensor_set(inp_cb0_embd, cb0_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                 cfg.code_pred_vocab_size * sizeof(float));

        output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }

    // Generate 14 more tokens autoregressively
#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    for (int step = 1; step < cfg.n_codebooks - 1; ++step) {
        int32_t n_past = step + 1;

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_step_graph(n_past, step);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor step graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_code = ggml_graph_get_tensor(gf, "inp_code");
        if (inp_code) {
            int32_t prev_code = output[step - 1];
            ggml_backend_tensor_set(inp_code, &prev_code, 0, sizeof(int32_t));
        }

        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t pos = n_past;
            ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                 cfg.code_pred_vocab_size * sizeof(float));

        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }
#ifdef QWEN3_TTS_TIMING
    if (timing_) timing_->t_code_pred_steps_ms += std::chrono::duration<double, std::milli>(clk::now() - t_steps_start).count();
#endif

    return true;
}

bool TTSTransformer::generate(const int32_t * text_tokens, int32_t n_tokens,
                               const float * speaker_embd, int32_t max_len,
                               std::vector<int32_t> & output,
                               int32_t language_id,
                               float repetition_penalty,
                               float temperature,
                               int32_t top_k,
                               int32_t speaker_token_id,
                               const int32_t * instruct_tokens,
                               int32_t n_instruct_tokens,
                               std::function<void(int, int)> progress_cb) {
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    tts_timing timing = {};
    auto t_gen_start = clk::now();
    auto t0 = t_gen_start, t1 = t_gen_start;
    timing_ = &timing;
#endif

    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for generation";
        return false;
    }
    if (max_len <= 0) {
        output.clear();
        return true;
    }

    const auto & cfg = model_.config;

    std::vector<float> prefill_embd;
    std::vector<float> trailing_text_hidden;
    std::vector<float> tts_pad_embed;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                             prefill_embd, trailing_text_hidden, tts_pad_embed,
                             speaker_token_id, instruct_tokens, n_instruct_tokens)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    const int32_t prefill_len = (int32_t)(prefill_embd.size() / cfg.hidden_size);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / cfg.hidden_size);

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx || state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) {
            return false;
        }
    }
    clear_kv_cache();
    std::vector<float> hidden_out;
    std::vector<float> logits;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    output.clear();
    output.reserve(max_len * cfg.n_codebooks);

    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = (cfg.codec_vocab_size >= 1024) ? (cfg.codec_vocab_size - 1024) : 0;

    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(cfg.hidden_size, 0.0f);
    std::vector<float> embd_row(cfg.hidden_size);

    for (int frame = 0; frame < max_len; ++frame) {
        // Suppress tokens in [codec_vocab_size - 1024, codec_vocab_size), except codec_eos_id
        for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i) {
            if (i != cfg.codec_eos_id) {
                logits[i] = -INFINITY;
            }
        }

        // Repetition penalty (HuggingFace style) on previously generated CB0 tokens
        if (repetition_penalty != 1.0f) {
            for (int32_t tok : generated_cb0_tokens) {
                if (tok >= 0 && tok < cfg.codec_vocab_size) {
                    if (logits[tok] > 0.0f) {
                        logits[tok] /= repetition_penalty;
                    } else {
                        logits[tok] *= repetition_penalty;
                    }
                }
            }
        }

        int32_t next_token;
        if (temperature <= 0.0f) {
            next_token = argmax(logits.data(), cfg.codec_vocab_size);
        } else {
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                logits[i] /= temperature;
            }

            if (top_k > 0 && top_k < cfg.codec_vocab_size) {
                std::vector<std::pair<float, int32_t>> scored(cfg.codec_vocab_size);
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    scored[i] = {logits[i], i};
                }
                std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                    [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                        return a.first > b.first;
                    });
                float threshold = scored[top_k - 1].first;
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    if (logits[i] < threshold) {
                        logits[i] = -INFINITY;
                    }
                }
            }

            float max_logit = *std::max_element(logits.data(), logits.data() + cfg.codec_vocab_size);
            double sum = 0.0;
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = expf(logits[i] - max_logit);
                sum += probs[i];
            }
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = (float)(probs[i] / sum);
            }

            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng_);
        }

        if (next_token == cfg.codec_eos_id) {
            break;
        }

        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0], codes_1_15, temperature, top_k)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_code_pred_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            frame_codes[cb] = codes_1_15[cb - 1];
        }

        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            output.push_back(frame_codes[cb]);
        }

        if (progress_cb) {
            progress_cb(frame + 1, max_len);
        }

#ifdef QWEN3_TTS_TIMING
        timing.n_frames = frame + 1;
#endif

        if (frame + 1 >= max_len) {
            break;
        }

        std::fill(step_embd.begin(), step_embd.end(), 0.0f);

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) {
            return false;
        }
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] = embd_row[h];
        }

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            int32_t code_token = frame_codes[cb];
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1], code_token, embd_row.data())) {
                return false;
            }
            for (int32_t h = 0; h < cfg.hidden_size; ++h) {
                step_embd[h] += embd_row[h];
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_embed_lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * cfg.hidden_size
            : tts_pad_embed.data();
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] += trailing_row[h];
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!forward_step(step_embd.data(), n_past, logits)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_talker_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        n_past++;
    }

#ifdef QWEN3_TTS_TIMING
    timing.t_generate_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_gen_start).count();
    timing_ = nullptr;
    const auto & t = timing;
    int nf = t.n_frames;
    fprintf(stderr, "\n=== Detailed Generation Timing (%d frames) ===\n", nf);
    fprintf(stderr, "\n  Prefill:\n");
    fprintf(stderr, "    Build graph:      %8.1f ms\n", t.t_prefill_build_ms);
    fprintf(stderr, "    Forward total:    %8.1f ms\n", t.t_prefill_forward_ms);
    fprintf(stderr, "      Graph build:    %8.1f ms\n", t.t_prefill_graph_build_ms);
    fprintf(stderr, "      Graph alloc:    %8.1f ms\n", t.t_prefill_graph_alloc_ms);
    fprintf(stderr, "      Compute:        %8.1f ms\n", t.t_prefill_compute_ms);
    fprintf(stderr, "      Data I/O:       %8.1f ms\n", t.t_prefill_data_ms);
    fprintf(stderr, "\n  Talker forward_step (total / per-frame):\n");
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_talker_forward_ms, nf > 0 ? t.t_talker_forward_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_build_ms, nf > 0 ? t.t_talker_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_alloc_ms, nf > 0 ? t.t_talker_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_talker_compute_ms, nf > 0 ? t.t_talker_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_talker_data_ms, nf > 0 ? t.t_talker_data_ms / nf : 0.0);
    fprintf(stderr, "\n  Code predictor (total / per-frame):\n");
    fprintf(stderr, "    Backend:          %s\n", use_coreml_code_predictor_ ? "CoreML (CPU+NE)" : "GGML");
    if (use_coreml_code_predictor_ && !coreml_code_predictor_path_.empty()) {
        fprintf(stderr, "    CoreML model:     %s\n", coreml_code_predictor_path_.c_str());
    }
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_ms, nf > 0 ? t.t_code_pred_ms / nf : 0.0);
    fprintf(stderr, "      Init/KV/embed:  %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_init_ms, nf > 0 ? t.t_code_pred_init_ms / nf : 0.0);
    fprintf(stderr, "      Prefill (2tok): %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_prefill_ms, nf > 0 ? t.t_code_pred_prefill_ms / nf : 0.0);
    fprintf(stderr, "      Steps (14):     %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_steps_ms, nf > 0 ? t.t_code_pred_steps_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_build_ms, nf > 0 ? t.t_code_pred_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_alloc_ms, nf > 0 ? t.t_code_pred_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_compute_ms, nf > 0 ? t.t_code_pred_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_data_ms, nf > 0 ? t.t_code_pred_data_ms / nf : 0.0);
    fprintf(stderr, "      CoreML total:   %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_coreml_ms, nf > 0 ? t.t_code_pred_coreml_ms / nf : 0.0);
    fprintf(stderr, "\n  Embed lookups:      %8.1f ms   (%.1f ms/frame)\n", t.t_embed_lookup_ms, nf > 0 ? t.t_embed_lookup_ms / nf : 0.0);
    double accounted = t.t_prefill_build_ms + t.t_prefill_forward_ms + t.t_talker_forward_ms + t.t_code_pred_ms + t.t_embed_lookup_ms;
    fprintf(stderr, "  Other/overhead:     %8.1f ms\n", t.t_generate_total_ms - accounted);
    fprintf(stderr, "  ─────────────────────────────────────────\n");
    fprintf(stderr, "  Total generate:     %8.1f ms\n", t.t_generate_total_ms);
    if (nf > 0) {
        fprintf(stderr, "  Throughput:         %8.1f ms/frame (%.1f frames/s)\n",
                t.t_generate_total_ms / nf, 1000.0 * nf / t.t_generate_total_ms);
    }
#endif

    return true;
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                              std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                         const float * audio_embd, int32_t n_audio,
                                         int32_t audio_start_pos, int32_t n_past,
                                         std::vector<float> & output) {
    (void)audio_embd;
    (void)n_audio;
    (void)audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts

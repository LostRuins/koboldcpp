#include <stdarg.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "gguf_reader.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"
#include "vocab.hpp"
#include "vocab_umt5.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "stable-diffusion.h"

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef SD_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef SD_USE_OPENCL
#include "ggml-opencl.h"
#endif

#define ST_HEADER_SIZE_LEN 8

static std::string format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

uint64_t read_u64(uint8_t* buffer) {
    // little endian
    uint64_t value = 0;
    value |= static_cast<int64_t>(buffer[7]) << 56;
    value |= static_cast<int64_t>(buffer[6]) << 48;
    value |= static_cast<int64_t>(buffer[5]) << 40;
    value |= static_cast<int64_t>(buffer[4]) << 32;
    value |= static_cast<int64_t>(buffer[3]) << 24;
    value |= static_cast<int64_t>(buffer[2]) << 16;
    value |= static_cast<int64_t>(buffer[1]) << 8;
    value |= static_cast<int64_t>(buffer[0]);
    return value;
}

int32_t read_int(uint8_t* buffer) {
    // little endian
    int value = 0;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

uint16_t read_short(uint8_t* buffer) {
    // little endian
    uint16_t value = 0;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

/*================================================= Preprocess ==================================================*/

std::string self_attn_names[] = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
};

const char* unused_tensors[] = {
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "cond_stage_model.model.text_projection",
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.0.model.logit_scale",
    "conditioner.embedders.1.model.logit_scale",
    "model.diffusion_model.time_embedding.cond_proj.weight",
    "unet.time_embedding.cond_proj.weight",
    "model_ema.decay",
    "model_ema.num_updates",
    "model_ema.diffusion_model",
    "embedding_manager",
    "denoiser.sigmas",
    "text_encoders.t5xxl.transformer.encoder.embed_tokens.weight",  // only used during training
};

bool is_unused_tensor(std::string name) {
    for (size_t i = 0; i < sizeof(unused_tensors) / sizeof(const char*); i++) {
        if (starts_with(name, unused_tensors[i])) {
            return true;
        }
    }
    return false;
}

std::unordered_map<std::string, std::string> open_clip_to_hf_clip_model = {
    {"model.ln_final.bias", "transformer.text_model.final_layer_norm.bias"},
    {"model.ln_final.weight", "transformer.text_model.final_layer_norm.weight"},
    {"model.positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"},
    {"model.token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"},
    {"model.text_projection", "transformer.text_model.text_projection"},
    {"model.visual.class_embedding", "transformer.vision_model.embeddings.class_embedding"},
    {"model.visual.conv1.weight", "transformer.vision_model.embeddings.patch_embedding.weight"},
    {"model.visual.ln_post.bias", "transformer.vision_model.post_layernorm.bias"},
    {"model.visual.ln_post.weight", "transformer.vision_model.post_layernorm.weight"},
    {"model.visual.ln_pre.bias", "transformer.vision_model.pre_layernorm.bias"},
    {"model.visual.ln_pre.weight", "transformer.vision_model.pre_layernorm.weight"},
    {"model.visual.positional_embedding", "transformer.vision_model.embeddings.position_embedding.weight"},
    {"model.visual.proj", "transformer.visual_projection.weight"},
};

std::unordered_map<std::string, std::string> open_clip_to_hk_clip_resblock = {
    {"attn.out_proj.bias", "self_attn.out_proj.bias"},
    {"attn.out_proj.weight", "self_attn.out_proj.weight"},
    {"ln_1.bias", "layer_norm1.bias"},
    {"ln_1.weight", "layer_norm1.weight"},
    {"ln_2.bias", "layer_norm2.bias"},
    {"ln_2.weight", "layer_norm2.weight"},
    {"mlp.c_fc.bias", "mlp.fc1.bias"},
    {"mlp.c_fc.weight", "mlp.fc1.weight"},
    {"mlp.c_proj.bias", "mlp.fc2.bias"},
    {"mlp.c_proj.weight", "mlp.fc2.weight"},
};

std::unordered_map<std::string, std::string> cond_model_name_map = {
    {"transformer.vision_model.pre_layrnorm.weight", "transformer.vision_model.pre_layernorm.weight"},
    {"transformer.vision_model.pre_layrnorm.bias", "transformer.vision_model.pre_layernorm.bias"},
};

std::unordered_map<std::string, std::string> vae_decoder_name_map = {
    {"first_stage_model.decoder.mid.attn_1.to_k.bias", "first_stage_model.decoder.mid.attn_1.k.bias"},
    {"first_stage_model.decoder.mid.attn_1.to_k.weight", "first_stage_model.decoder.mid.attn_1.k.weight"},
    {"first_stage_model.decoder.mid.attn_1.to_out.0.bias", "first_stage_model.decoder.mid.attn_1.proj_out.bias"},
    {"first_stage_model.decoder.mid.attn_1.to_out.0.weight", "first_stage_model.decoder.mid.attn_1.proj_out.weight"},
    {"first_stage_model.decoder.mid.attn_1.to_q.bias", "first_stage_model.decoder.mid.attn_1.q.bias"},
    {"first_stage_model.decoder.mid.attn_1.to_q.weight", "first_stage_model.decoder.mid.attn_1.q.weight"},
    {"first_stage_model.decoder.mid.attn_1.to_v.bias", "first_stage_model.decoder.mid.attn_1.v.bias"},
    {"first_stage_model.decoder.mid.attn_1.to_v.weight", "first_stage_model.decoder.mid.attn_1.v.weight"},
};

std::unordered_map<std::string, std::string> pmid_v2_name_map = {
    {"pmid.qformer_perceiver.perceiver_resampler.layers.0.1.1.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.0.1.1.fc1.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.0.1.3.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.0.1.1.fc2.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.1.1.1.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.1.1.1.fc1.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.1.1.3.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.1.1.1.fc2.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.2.1.1.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.2.1.1.fc1.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.2.1.3.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.2.1.1.fc2.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.3.1.1.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.3.1.1.fc1.weight"},
    {"pmid.qformer_perceiver.perceiver_resampler.layers.3.1.3.weight",
     "pmid.qformer_perceiver.perceiver_resampler.layers.3.1.1.fc2.weight"},
    {"pmid.qformer_perceiver.token_proj.0.bias",
     "pmid.qformer_perceiver.token_proj.fc1.bias"},
    {"pmid.qformer_perceiver.token_proj.2.bias",
     "pmid.qformer_perceiver.token_proj.fc2.bias"},
    {"pmid.qformer_perceiver.token_proj.0.weight",
     "pmid.qformer_perceiver.token_proj.fc1.weight"},
    {"pmid.qformer_perceiver.token_proj.2.weight",
     "pmid.qformer_perceiver.token_proj.fc2.weight"},
};

std::string convert_cond_model_name(const std::string& name) {
    std::string new_name = name;
    std::string prefix;
    if (contains(new_name, ".enc.")) {
        // llama.cpp naming convention for T5
        size_t pos = new_name.find(".enc.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 5, ".encoder.");
        }
        pos = new_name.find("blk.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 4, "block.");
        }
        pos = new_name.find("output_norm.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 12, "final_layer_norm.");
        }
        pos = new_name.find("attn_k.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 7, "layer.0.SelfAttention.k.");
        }
        pos = new_name.find("attn_v.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 7, "layer.0.SelfAttention.v.");
        }
        pos = new_name.find("attn_o.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 7, "layer.0.SelfAttention.o.");
        }
        pos = new_name.find("attn_q.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 7, "layer.0.SelfAttention.q.");
        }
        pos = new_name.find("attn_norm.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 10, "layer.0.layer_norm.");
        }
        pos = new_name.find("ffn_norm.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 9, "layer.1.layer_norm.");
        }
        pos = new_name.find("ffn_up.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 7, "layer.1.DenseReluDense.wi_1.");
        }
        pos = new_name.find("ffn_down.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 9, "layer.1.DenseReluDense.wo.");
        }
        pos = new_name.find("ffn_gate.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 9, "layer.1.DenseReluDense.wi_0.");
        }
        pos = new_name.find("attn_rel_b.");
        if (pos != std::string::npos) {
            new_name.replace(pos, 11, "layer.0.SelfAttention.relative_attention_bias.");
        }
    } else if (name == "text_encoders.t5xxl.transformer.token_embd.weight") {
        new_name = "text_encoders.t5xxl.transformer.shared.weight";
    }

    if (starts_with(new_name, "conditioner.embedders.0.open_clip.")) {
        prefix   = "cond_stage_model.";
        new_name = new_name.substr(strlen("conditioner.embedders.0.open_clip."));
    } else if (starts_with(new_name, "conditioner.embedders.0.")) {
        prefix   = "cond_stage_model.";
        new_name = new_name.substr(strlen("conditioner.embedders.0."));
    } else if (starts_with(new_name, "conditioner.embedders.1.")) {
        prefix   = "cond_stage_model.1.";
        new_name = new_name.substr(strlen("conditioner.embedders.0."));
    } else if (starts_with(new_name, "cond_stage_model.")) {
        prefix   = "cond_stage_model.";
        new_name = new_name.substr(strlen("cond_stage_model."));
    } else if (ends_with(new_name, "vision_model.visual_projection.weight")) {
        prefix   = new_name.substr(0, new_name.size() - strlen("vision_model.visual_projection.weight"));
        new_name = prefix + "visual_projection.weight";
        return new_name;
    } else if (ends_with(new_name, "transformer.text_projection.weight")) {
        prefix   = new_name.substr(0, new_name.size() - strlen("transformer.text_projection.weight"));
        new_name = prefix + "transformer.text_model.text_projection";
        return new_name;
    } else {
        return new_name;
    }

    if (open_clip_to_hf_clip_model.find(new_name) != open_clip_to_hf_clip_model.end()) {
        new_name = open_clip_to_hf_clip_model[new_name];
    }

    if (cond_model_name_map.find(new_name) != cond_model_name_map.end()) {
        new_name = cond_model_name_map[new_name];
    }

    std::string open_clip_resblock_prefix = "model.transformer.resblocks.";
    std::string hf_clip_resblock_prefix   = "transformer.text_model.encoder.layers.";

    auto replace_suffix = [&]() {
        if (new_name.find(open_clip_resblock_prefix) == 0) {
            std::string remain = new_name.substr(open_clip_resblock_prefix.length());
            std::string idx    = remain.substr(0, remain.find("."));
            std::string suffix = remain.substr(idx.length() + 1);

            if (suffix == "attn.in_proj_weight" || suffix == "attn.in_proj_bias") {
                new_name = hf_clip_resblock_prefix + idx + "." + suffix;
            } else if (open_clip_to_hk_clip_resblock.find(suffix) != open_clip_to_hk_clip_resblock.end()) {
                std::string new_suffix = open_clip_to_hk_clip_resblock[suffix];
                new_name               = hf_clip_resblock_prefix + idx + "." + new_suffix;
            }
        }
    };

    replace_suffix();

    open_clip_resblock_prefix = "model.visual.transformer.resblocks.";
    hf_clip_resblock_prefix   = "transformer.vision_model.encoder.layers.";

    replace_suffix();

    return prefix + new_name;
}

std::string convert_vae_decoder_name(const std::string& name) {
    if (vae_decoder_name_map.find(name) != vae_decoder_name_map.end()) {
        return vae_decoder_name_map[name];
    }
    return name;
}

std::string convert_pmid_v2_name(const std::string& name) {
    if (pmid_v2_name_map.find(name) != pmid_v2_name_map.end()) {
        return pmid_v2_name_map[name];
    }
    return name;
}

/* If not a SDXL LoRA the unet" prefix will have already been replaced by this
 * point and "te2" and "te1" don't seem to appear in non-SDXL only "te_" */
std::string convert_sdxl_lora_name(std::string tensor_name) {
    const std::pair<std::string, std::string> sdxl_lora_name_lookup[] = {
        {"unet", "model_diffusion_model"},
        {"te2", "cond_stage_model_1_transformer"},
        {"te1", "cond_stage_model_transformer"},
        {"text_encoder_2", "cond_stage_model_1_transformer"},
        {"text_encoder", "cond_stage_model_transformer"},
    };
    for (auto& pair_i : sdxl_lora_name_lookup) {
        if (tensor_name.compare(0, pair_i.first.length(), pair_i.first) == 0) {
            tensor_name = std::regex_replace(tensor_name, std::regex(pair_i.first), pair_i.second);
            break;
        }
    }
    return tensor_name;
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> suffix_conversion_underline = {
    {
        "attentions",
        {
            {"to_k", "k"},
            {"to_q", "q"},
            {"to_v", "v"},
            {"to_out_0", "proj_out"},
            {"group_norm", "norm"},
            {"key", "k"},
            {"query", "q"},
            {"value", "v"},
            {"proj_attn", "proj_out"},
        },
    },
    {
        "resnets",
        {
            {"conv1", "in_layers_2"},
            {"conv2", "out_layers_3"},
            {"norm1", "in_layers_0"},
            {"norm2", "out_layers_0"},
            {"time_emb_proj", "emb_layers_1"},
            {"conv_shortcut", "skip_connection"},
        },
    },
};

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> suffix_conversion_dot = {
    {
        "attentions",
        {
            {"to_k", "k"},
            {"to_q", "q"},
            {"to_v", "v"},
            {"to_out.0", "proj_out"},
            {"group_norm", "norm"},
            {"key", "k"},
            {"query", "q"},
            {"value", "v"},
            {"proj_attn", "proj_out"},
        },
    },
    {
        "resnets",
        {
            {"conv1", "in_layers.2"},
            {"conv2", "out_layers.3"},
            {"norm1", "in_layers.0"},
            {"norm2", "out_layers.0"},
            {"time_emb_proj", "emb_layers.1"},
            {"conv_shortcut", "skip_connection"},
        },
    },
};

std::string convert_diffusers_name_to_compvis(std::string key, char seq) {
    std::vector<std::string> m;

    auto match = [](std::vector<std::string>& match_list, const std::regex& regex, const std::string& key) {
        auto r = std::smatch{};
        if (!std::regex_match(key, r, regex)) {
            return false;
        }

        match_list.clear();
        for (size_t i = 1; i < r.size(); ++i) {
            match_list.push_back(r.str(i));
        }
        return true;
    };

    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> suffix_conversion;
    if (seq == '_') {
        suffix_conversion = suffix_conversion_underline;
    } else {
        suffix_conversion = suffix_conversion_dot;
    }

    auto get_converted_suffix = [&suffix_conversion](const std::string& outer_key, const std::string& inner_key) {
        auto outer_iter = suffix_conversion.find(outer_key);
        if (outer_iter != suffix_conversion.end()) {
            auto inner_iter = outer_iter->second.find(inner_key);
            if (inner_iter != outer_iter->second.end()) {
                return inner_iter->second;
            }
        }
        return inner_key;
    };

    // convert attn to out
    if (ends_with(key, "to_out")) {
        key += format("%c0", seq);
    }

    // unet
    if (match(m, std::regex(format("unet%cconv_in(.*)", seq)), key)) {
        return format("model%cdiffusion_model%cinput_blocks%c0%c0", seq, seq, seq, seq) + m[0];
    }

    if (match(m, std::regex(format("unet%cconv%cout(.*)", seq, seq)), key)) {
        return format("model%cdiffusion_model%cout%c2", seq, seq, seq) + m[0];
    }

    if (match(m, std::regex(format("unet%cconv_norm_out(.*)", seq)), key)) {
        return format("model%cdiffusion_model%cout%c0", seq, seq, seq) + m[0];
    }

    if (match(m, std::regex(format("unet%ctime_embedding%clinear_(\\d+)(.*)", seq, seq)), key)) {
        return format("model%cdiffusion_model%ctime_embed%c", seq, seq, seq) + std::to_string(std::stoi(m[0]) * 2 - 2) + m[1];
    }

    if (match(m, std::regex(format("unet%cadd_embedding%clinear_(\\d+)(.*)", seq, seq)), key)) {
        return format("model%cdiffusion_model%clabel_emb%c0%c", seq, seq, seq, seq) + std::to_string(std::stoi(m[0]) * 2 - 2) + m[1];
    }

    if (match(m, std::regex(format("unet%cdown_blocks%c(\\d+)%c(attentions|resnets)%c(\\d+)%c(.+)", seq, seq, seq, seq, seq)), key)) {
        std::string suffix = get_converted_suffix(m[1], m[3]);
        // LOG_DEBUG("%s %s %s %s", m[0].c_str(), m[1].c_str(), m[2].c_str(), m[3].c_str());
        return format("model%cdiffusion_model%cinput_blocks%c", seq, seq, seq) + std::to_string(1 + std::stoi(m[0]) * 3 + std::stoi(m[2])) + seq +
               (m[1] == "attentions" ? "1" : "0") + seq + suffix;
    }

    if (match(m, std::regex(format("unet%cmid_block%c(attentions|resnets)%c(\\d+)%c(.+)", seq, seq, seq, seq)), key)) {
        std::string suffix = get_converted_suffix(m[0], m[2]);
        return format("model%cdiffusion_model%cmiddle_block%c", seq, seq, seq) + (m[0] == "attentions" ? "1" : std::to_string(std::stoi(m[1]) * 2)) +
               seq + suffix;
    }

    if (match(m, std::regex(format("unet%cup_blocks%c(\\d+)%c(attentions|resnets)%c(\\d+)%c(.+)", seq, seq, seq, seq, seq)), key)) {
        std::string suffix = get_converted_suffix(m[1], m[3]);
        return format("model%cdiffusion_model%coutput_blocks%c", seq, seq, seq) + std::to_string(std::stoi(m[0]) * 3 + std::stoi(m[2])) + seq +
               (m[1] == "attentions" ? "1" : "0") + seq + suffix;
    }

    if (match(m, std::regex(format("unet%cdown_blocks%c(\\d+)%cdownsamplers%c0%cconv", seq, seq, seq, seq, seq)), key)) {
        return format("model%cdiffusion_model%cinput_blocks%c", seq, seq, seq) + std::to_string(3 + std::stoi(m[0]) * 3) + seq + "0" + seq + "op";
    }

    if (match(m, std::regex(format("unet%cup_blocks%c(\\d+)%cupsamplers%c0%cconv", seq, seq, seq, seq, seq)), key)) {
        return format("model%cdiffusion_model%coutput_blocks%c", seq, seq, seq) + std::to_string(2 + std::stoi(m[0]) * 3) + seq +
               (std::stoi(m[0]) > 0 ? "2" : "1") + seq + "conv";
    }

    // clip
    if (match(m, std::regex(format("te%ctext_model%cencoder%clayers%c(\\d+)%c(.+)", seq, seq, seq, seq, seq)), key)) {
        return format("cond_stage_model%ctransformer%ctext_model%cencoder%clayers%c", seq, seq, seq, seq, seq) + m[0] + seq + m[1];
    }

    if (match(m, std::regex(format("te%ctext_model(.*)", seq)), key)) {
        return format("cond_stage_model%ctransformer%ctext_model", seq, seq) + m[0];
    }

    // clip-g
    if (match(m, std::regex(format("te%c1%ctext_model%cencoder%clayers%c(\\d+)%c(.+)", seq, seq, seq, seq, seq, seq)), key)) {
        return format("cond_stage_model%c1%ctransformer%ctext_model%cencoder%clayers%c", seq, seq, seq, seq, seq, seq) + m[0] + seq + m[1];
    }

    if (match(m, std::regex(format("te%c1%ctext_model(.*)", seq, seq)), key)) {
        return format("cond_stage_model%c1%ctransformer%ctext_model", seq, seq, seq) + m[0];
    }

    if (match(m, std::regex(format("te%c1%ctext_projection", seq, seq)), key)) {
        return format("cond_stage_model%c1%ctransformer%ctext_model%ctext_projection", seq, seq, seq, seq);
    }

    // vae
    if (match(m, std::regex(format("vae%c(.*)%cconv_norm_out(.*)", seq, seq)), key)) {
        return format("first_stage_model%c%s%cnorm_out%s", seq, m[0].c_str(), seq, m[1].c_str());
    }

    if (match(m, std::regex(format("vae%c(.*)%cmid_block%c(attentions|resnets)%c(\\d+)%c(.+)", seq, seq, seq, seq, seq)), key)) {
        std::string suffix;
        std::string block_name;
        if (m[1] == "attentions") {
            block_name = "attn";
            suffix     = get_converted_suffix(m[1], m[3]);
        } else {
            block_name = "block";
            suffix     = m[3];
        }
        return format("first_stage_model%c%s%cmid%c%s_%d%c%s",
                      seq, m[0].c_str(), seq, seq, block_name.c_str(), std::stoi(m[2]) + 1, seq, suffix.c_str());
    }

    if (match(m, std::regex(format("vae%c(.*)%cup_blocks%c(\\d+)%cresnets%c(\\d+)%c(.+)", seq, seq, seq, seq, seq, seq)), key)) {
        std::string suffix = m[3];
        if (suffix == "conv_shortcut") {
            suffix = "nin_shortcut";
        }
        return format("first_stage_model%c%s%cup%c%d%cblock%c%s%c%s",
                      seq, m[0].c_str(), seq, seq, 3 - std::stoi(m[1]), seq, seq, m[2].c_str(), seq, suffix.c_str());
    }

    if (match(m, std::regex(format("vae%c(.*)%cdown_blocks%c(\\d+)%cdownsamplers%c0%cconv", seq, seq, seq, seq, seq, seq)), key)) {
        return format("first_stage_model%c%s%cdown%c%d%cdownsample%cconv",
                      seq, m[0].c_str(), seq, seq, std::stoi(m[1]), seq, seq);
    }

    if (match(m, std::regex(format("vae%c(.*)%cdown_blocks%c(\\d+)%cresnets%c(\\d+)%c(.+)", seq, seq, seq, seq, seq, seq)), key)) {
        std::string suffix = m[3];
        if (suffix == "conv_shortcut") {
            suffix = "nin_shortcut";
        }
        return format("first_stage_model%c%s%cdown%c%d%cblock%c%s%c%s",
                      seq, m[0].c_str(), seq, seq, std::stoi(m[1]), seq, seq, m[2].c_str(), seq, suffix.c_str());
    }

    if (match(m, std::regex(format("vae%c(.*)%cup_blocks%c(\\d+)%cupsamplers%c0%cconv", seq, seq, seq, seq, seq, seq)), key)) {
        return format("first_stage_model%c%s%cup%c%d%cupsample%cconv",
                      seq, m[0].c_str(), seq, seq, 3 - std::stoi(m[1]), seq, seq);
    }

    if (match(m, std::regex(format("vae%c(.*)", seq)), key)) {
        return format("first_stage_model%c", seq) + m[0];
    }

    return key;
}

std::string convert_tensor_name(std::string name) {
    if (starts_with(name, "diffusion_model")) {
        name = "model." + name;
    }
    // size_t pos = name.find("lora_A");
    // if (pos != std::string::npos) {
    //     name.replace(pos, strlen("lora_A"), "lora_up");
    // }
    // pos = name.find("lora_B");
    // if (pos != std::string::npos) {
    //     name.replace(pos, strlen("lora_B"), "lora_down");
    // }
    std::string new_name = name;
    if (starts_with(name, "cond_stage_model.") || starts_with(name, "conditioner.embedders.") || starts_with(name, "text_encoders.") || ends_with(name, ".vision_model.visual_projection.weight")) {
        new_name = convert_cond_model_name(name);
    } else if (starts_with(name, "first_stage_model.decoder")) {
        new_name = convert_vae_decoder_name(name);
    } else if (starts_with(name, "pmid.qformer_perceiver")) {
        new_name = convert_pmid_v2_name(name);
    } else if (starts_with(name, "control_model.")) {  // for controlnet pth models
        size_t pos = name.find('.');
        if (pos != std::string::npos) {
            new_name = name.substr(pos + 1);
        }
    } else if (starts_with(name, "lora_")) {  // for lora
        size_t pos = name.find('.');
        if (pos != std::string::npos) {
            std::string name_without_network_parts = name.substr(5, pos - 5);
            std::string network_part               = name.substr(pos + 1);

            // LOG_DEBUG("%s %s", name_without_network_parts.c_str(), network_part.c_str());
            std::string new_key = convert_diffusers_name_to_compvis(name_without_network_parts, '_');
            /* For dealing with the new SDXL LoRA tensor naming convention */
            new_key = convert_sdxl_lora_name(new_key);

            if (new_key.empty()) {
                new_name = name;
            } else {
                new_name = "lora." + new_key + "." + network_part;
            }
        } else {
            new_name = name;
        }
    } else if (ends_with(name, ".diff") || ends_with(name, ".diff_b")) {
        new_name = "lora." + name;
    } else if (contains(name, "lora_up") || contains(name, "lora_down") ||
               contains(name, "lora.up") || contains(name, "lora.down") ||
               contains(name, "lora_linear") || ends_with(name, ".alpha")) {
        size_t pos = new_name.find(".processor");
        if (pos != std::string::npos) {
            new_name.replace(pos, strlen(".processor"), "");
        }
        // if (starts_with(new_name, "transformer.transformer_blocks") || starts_with(new_name, "transformer.single_transformer_blocks")) {
        //     new_name = "model.diffusion_model." + new_name;
        // }
        if (ends_with(name, ".alpha")) {
            pos = new_name.rfind("alpha");
        } else {
            pos = new_name.rfind("lora");
        }
        if (pos != std::string::npos) {
            std::string name_without_network_parts = new_name.substr(0, pos - 1);
            std::string network_part               = new_name.substr(pos);
            // LOG_DEBUG("%s %s", name_without_network_parts.c_str(), network_part.c_str());
            std::string new_key = convert_diffusers_name_to_compvis(name_without_network_parts, '.');
            new_key             = convert_sdxl_lora_name(new_key);
            replace_all_chars(new_key, '.', '_');
            size_t npos = network_part.rfind("_linear_layer");
            if (npos != std::string::npos) {
                network_part.replace(npos, strlen("_linear_layer"), "");
            }
            if (starts_with(network_part, "lora.")) {
                network_part = "lora_" + network_part.substr(5);
            }
            if (new_key.size() > 0) {
                new_name = "lora." + new_key + "." + network_part;
            }
            // LOG_DEBUG("new name: %s", new_name.c_str());
        }
    } else if (starts_with(name, "unet") || starts_with(name, "vae") || starts_with(name, "te")) {  // for diffuser
        size_t pos = name.find_last_of('.');
        if (pos != std::string::npos) {
            std::string name_without_network_parts = name.substr(0, pos);
            std::string network_part               = name.substr(pos + 1);
            // LOG_DEBUG("%s %s", name_without_network_parts.c_str(), network_part.c_str());
            std::string new_key = convert_diffusers_name_to_compvis(name_without_network_parts, '.');
            if (new_key.empty()) {
                new_name = name;
            } else if (new_key == "cond_stage_model.1.transformer.text_model.text_projection") {
                new_name = new_key;
            } else {
                new_name = new_key + "." + network_part;
            }
        } else {
            new_name = name;
        }
    } else {
        new_name = name;
    }
    // if (new_name != name) {
    //     LOG_DEBUG("%s => %s", name.c_str(), new_name.c_str());
    // }
    return new_name;
}

void add_preprocess_tensor_storage_types(String2GGMLType& tensor_storages_types, std::string name, enum ggml_type type) {
    std::string new_name = convert_tensor_name(name);

    if (new_name.find("cond_stage_model") != std::string::npos && ends_with(new_name, "attn.in_proj_weight")) {
        size_t prefix_size                                        = new_name.find("attn.in_proj_weight");
        std::string prefix                                        = new_name.substr(0, prefix_size);
        tensor_storages_types[prefix + "self_attn.q_proj.weight"] = type;
        tensor_storages_types[prefix + "self_attn.k_proj.weight"] = type;
        tensor_storages_types[prefix + "self_attn.v_proj.weight"] = type;
    } else if (new_name.find("cond_stage_model") != std::string::npos && ends_with(new_name, "attn.in_proj_bias")) {
        size_t prefix_size                                      = new_name.find("attn.in_proj_bias");
        std::string prefix                                      = new_name.substr(0, prefix_size);
        tensor_storages_types[prefix + "self_attn.q_proj.bias"] = type;
        tensor_storages_types[prefix + "self_attn.k_proj.bias"] = type;
        tensor_storages_types[prefix + "self_attn.v_proj.bias"] = type;
    } else {
        tensor_storages_types[new_name] = type;
    }
}

void preprocess_tensor(TensorStorage tensor_storage,
                       std::vector<TensorStorage>& processed_tensor_storages) {
    std::vector<TensorStorage> result;
    std::string new_name = convert_tensor_name(tensor_storage.name);

    // convert unet transformer linear to conv2d 1x1
    if (starts_with(new_name, "model.diffusion_model.") &&
        (ends_with(new_name, "proj_in.weight") || ends_with(new_name, "proj_out.weight"))) {
        tensor_storage.unsqueeze();
    }

    // convert vae attn block linear to conv2d 1x1
    if (starts_with(new_name, "first_stage_model.") && new_name.find("attn_1") != std::string::npos) {
        tensor_storage.unsqueeze();
    }

    // wan vae
    if (ends_with(new_name, "gamma")) {
        tensor_storage.reverse_ne();
        tensor_storage.n_dims = 1;
        tensor_storage.reverse_ne();
    }

    tensor_storage.name = new_name;

    if (new_name.find("cond_stage_model") != std::string::npos &&
        ends_with(new_name, "attn.in_proj_weight")) {
        size_t prefix_size = new_name.find("attn.in_proj_weight");
        std::string prefix = new_name.substr(0, prefix_size);

        std::vector<TensorStorage> chunks = tensor_storage.chunk(3);
        chunks[0].name                    = prefix + "self_attn.q_proj.weight";
        chunks[1].name                    = prefix + "self_attn.k_proj.weight";
        chunks[2].name                    = prefix + "self_attn.v_proj.weight";

        processed_tensor_storages.insert(processed_tensor_storages.end(), chunks.begin(), chunks.end());

    } else if (new_name.find("cond_stage_model") != std::string::npos &&
               ends_with(new_name, "attn.in_proj_bias")) {
        size_t prefix_size = new_name.find("attn.in_proj_bias");
        std::string prefix = new_name.substr(0, prefix_size);

        std::vector<TensorStorage> chunks = tensor_storage.chunk(3);
        chunks[0].name                    = prefix + "self_attn.q_proj.bias";
        chunks[1].name                    = prefix + "self_attn.k_proj.bias";
        chunks[2].name                    = prefix + "self_attn.v_proj.bias";

        processed_tensor_storages.insert(processed_tensor_storages.end(), chunks.begin(), chunks.end());
    } else {
        processed_tensor_storages.push_back(tensor_storage);
    }
}

float bf16_to_f32(uint16_t bfloat16) {
    uint32_t val_bits = (static_cast<uint32_t>(bfloat16) << 16);
    return *reinterpret_cast<float*>(&val_bits);
}

uint16_t f8_e4m3_to_f16(uint8_t f8) {
    // do we need to support uz?

    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return ggml_fp32_to_fp16(-NAN);
    } else if (f8 == 0x7f) {
        return ggml_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;

            // yes, 2 times
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }

            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }

    return ggml_fp32_to_fp16(*reinterpret_cast<const float*>(&result));
}

uint16_t f8_e5m2_to_f16(uint8_t fp8) {
    uint8_t sign     = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 2) & 0x1F;
    uint8_t mantissa = fp8 & 0x3;

    uint16_t fp16_sign = sign << 15;
    uint16_t fp16_exponent;
    uint16_t fp16_mantissa;

    if (exponent == 0 && mantissa == 0) {  // zero
        return fp16_sign;
    }

    if (exponent == 0x1F) {  // NAN and INF
        fp16_exponent = 0x1F;
        fp16_mantissa = mantissa ? (mantissa << 8) : 0;
        return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
    }

    if (exponent == 0) {  // subnormal numbers
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
        return fp16_sign | fp16_mantissa;
    }

    // normal numbers
    int16_t true_exponent = (int16_t)exponent - 15 + 15;
    if (true_exponent <= 0) {
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
    } else if (true_exponent >= 0x1F) {
        fp16_exponent = 0x1F;
        fp16_mantissa = 0;
    } else {
        fp16_exponent = (uint16_t)true_exponent;
        fp16_mantissa = mantissa << 8;
    }

    return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
}

void bf16_to_f32_vec(uint16_t* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = bf16_to_f32(src[i]);
    }
}

void f8_e4m3_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e4m3_to_f16(src[i]);
    }
}

void f8_e5m2_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e5m2_to_f16(src[i]);
    }
}

void f64_to_f32_vec(double* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

void i64_to_i32_vec(int64_t* src, int32_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (int32_t)src[i];
    }
}

void convert_tensor(void* src,
                    ggml_type src_type,
                    void* dst,
                    ggml_type dst_type,
                    int nrows,
                    int n_per_row) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_get_type_traits(src_type);
            if (qtype->to_float == NULL) {
                throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                                ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                            ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype->to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}

/*================================================= ModelLoader ==================================================*/

// ported from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
std::map<char, int> unicode_to_byte() {
    std::map<int, char> byte_to_unicode;

    // List of utf-8 byte ranges
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }

    for (int b = 49825; b <= 49836; ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }

    for (int b = 49838; b <= 50111; ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }
    // printf("%d %d %d %d\n", static_cast<int>('¡'), static_cast<int>('¬'), static_cast<int>('®'), static_cast<int>('ÿ'));
    // exit(1);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_to_unicode.find(b) == byte_to_unicode.end()) {
            byte_to_unicode[b] = static_cast<char>(256 + n);
            n++;
        }
    }

    // byte_encoder = bytes_to_unicode()
    // byte_decoder = {v: k for k, v in byte_encoder.items()}
    std::map<char, int> byte_decoder;

    for (const auto& entry : byte_to_unicode) {
        byte_decoder[entry.second] = entry.first;
    }

    byte_to_unicode.clear();

    return byte_decoder;
}

bool is_zip_file(const std::string& file_path) {
    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        return false;
    }
    zip_close(zip);
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    #ifdef _WIN32
        std::filesystem::path fpath = std::filesystem::u8path(file_path);
    #else
        std::filesystem::path fpath = std::filesystem::path(file_path);
    #endif
    std::ifstream file(fpath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool is_safetensors_file(const std::string& file_path) {
    #ifdef _WIN32
        std::filesystem::path fpath = std::filesystem::u8path(file_path);
    #else
        std::filesystem::path fpath = std::filesystem::path(file_path);
    #endif
    std::ifstream file(fpath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_ || header_size_ <= 2) {
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        return false;
    }
    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());
    if (header_.is_discarded()) {
        return false;
    }
    return true;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (is_directory(file_path)) {
        LOG_INFO("load %s using diffusers format", file_path.c_str());
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        LOG_INFO("load %s using safetensors format", file_path.c_str());
        return init_from_safetensors_file(file_path, prefix);
    //disable ckpt loading
    // } else if (is_zip_file(file_path)) {
    //     LOG_INFO("load %s using checkpoint format", file_path.c_str());
    //     return init_from_ckpt_file(file_path, prefix);
    } else {
        LOG_WARN("unknown format %s", file_path.c_str());
        return false;
    }
}

/*================================================= GGUFModelLoader ==================================================*/

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    gguf_context* ctx_gguf_ = NULL;
    ggml_context* ctx_meta_ = NULL;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        LOG_ERROR("failed to open '%s' with gguf_init_from_file. Try to open it with GGUFReader.", file_path.c_str());
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            LOG_ERROR("failed to open '%s' with GGUFReader.", file_path.c_str());
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            std::string name = gguf_tensor_info.name;
            if (!starts_with(name, prefix)) {
                name = prefix + name;
            }

            TensorStorage tensor_storage(
                name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                gguf_tensor_info.shape.size(),
                file_index,
                data_offset + gguf_tensor_info.offset);

            // LOG_DEBUG("%s %s", name.c_str(), tensor_storage.to_string().c_str());

            tensor_storages.push_back(tensor_storage);
            add_preprocess_tensor_storage_types(tensor_storages_types, tensor_storage.name, tensor_storage.type);
        }

        return true;
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf_);

    size_t total_size  = 0;
    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name          = gguf_get_tensor_name(ctx_gguf_, i);
        struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset             = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        // LOG_DEBUG("%s", name.c_str());

        if (!starts_with(name, prefix)) {
            name = prefix + name;
        }

        TensorStorage tensor_storage(name, dummy->type, dummy->ne, ggml_n_dims(dummy), file_index, offset);

        GGML_ASSERT(ggml_nbytes(dummy) == tensor_storage.nbytes());

        tensor_storages.push_back(tensor_storage);
        add_preprocess_tensor_storage_types(tensor_storages_types, tensor_storage.name, tensor_storage.type);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

/*================================================= SafeTensorsModelLoader ==================================================*/

ggml_type str_to_ggml_type(const std::string& dtype) {
    ggml_type ttype = GGML_TYPE_COUNT;
    if (dtype == "F16") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "BF16") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F32") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F64") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F8_E4M3") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "F8_E5M2") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "I64") {
        ttype = GGML_TYPE_I32;
    }
    return ttype;
}

// https://huggingface.co/docs/safetensors/index
bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;
    #ifdef _WIN32
        std::filesystem::path fpath = std::filesystem::u8path(file_path);
    #else
        std::filesystem::path fpath = std::filesystem::path(file_path);
    #endif
    std::ifstream file(fpath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        LOG_ERROR("read safetensors header size failed: '%s'", file_path.c_str());
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        LOG_ERROR("read safetensors header failed: '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());

    for (auto& item : header_.items()) {
        std::string name           = item.key();
        nlohmann::json tensor_info = item.value();
        // LOG_DEBUG("%s %s\n", name.c_str(), tensor_info.dump().c_str());

        if (name == "__metadata__") {
            continue;
        }

        if (is_unused_tensor(name)) {
            continue;
        }

        std::string dtype    = tensor_info["dtype"];
        nlohmann::json shape = tensor_info["shape"];

        if (dtype == "U8") {
            continue;
        }

        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();

        ggml_type type = str_to_ggml_type(dtype);
        if (type == GGML_TYPE_COUNT) {
            LOG_ERROR("unsupported dtype '%s' (tensor '%s')", dtype.c_str(), name.c_str());
            return false;
        }

        if (shape.size() > SD_MAX_DIMS) {
            LOG_ERROR("invalid tensor '%s'", name.c_str());
            return false;
        }

        int n_dims              = (int)shape.size();
        int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape[i].get<int64_t>();
        }

        if (n_dims == 5) {
            n_dims = 4;
            ne[0]  = ne[0] * ne[1];
            ne[1]  = ne[2];
            ne[2]  = ne[3];
            ne[3]  = ne[4];
        }

        // ggml_n_dims returns 1 for scalars
        if (n_dims == 0) {
            n_dims = 1;
        }

        if (!starts_with(name, prefix)) {
            name = prefix + name;
        }

        TensorStorage tensor_storage(name, type, ne, n_dims, file_index, ST_HEADER_SIZE_LEN + header_size_ + begin);
        tensor_storage.reverse_ne();

        size_t tensor_data_size = end - begin;

        if (dtype == "BF16") {
            tensor_storage.is_bf16 = true;
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E4M3") {
            tensor_storage.is_f8_e4m3 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E5M2") {
            tensor_storage.is_f8_e5m2 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F64") {
            tensor_storage.is_f64 = true;
            // f64 -> f32
            GGML_ASSERT(tensor_storage.nbytes() * 2 == tensor_data_size);
        } else if (dtype == "I64") {
            tensor_storage.is_i64 = true;
            // i64 -> i32
            GGML_ASSERT(tensor_storage.nbytes() * 2 == tensor_data_size);
        } else {
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size);
        }

        tensor_storages.push_back(tensor_storage);
        add_preprocess_tensor_storage_types(tensor_storages_types, tensor_storage.name, tensor_storage.type);

        // LOG_DEBUG("%s %s", tensor_storage.to_string().c_str(), dtype.c_str());
    }

    return true;
}

/*================================================= DiffusersModelLoader ==================================================*/

bool ModelLoader::init_from_diffusers_file(const std::string& file_path, const std::string& prefix) {
    std::string unet_path   = path_join(file_path, "unet/diffusion_pytorch_model.safetensors");
    std::string vae_path    = path_join(file_path, "vae/diffusion_pytorch_model.safetensors");
    std::string clip_path   = path_join(file_path, "text_encoder/model.safetensors");
    std::string clip_g_path = path_join(file_path, "text_encoder_2/model.safetensors");

    if (!init_from_safetensors_file(unet_path, "unet.")) {
        return false;
    }
    for (auto ts : tensor_storages) {
        if (ts.name.find("add_embedding") != std::string::npos || ts.name.find("label_emb") != std::string::npos) {
            // probably SDXL
            LOG_DEBUG("Fixing name for SDXL output blocks.2.2");
            for (auto& tensor_storage : tensor_storages) {
                int len  = 34;
                auto pos = tensor_storage.name.find("unet.up_blocks.0.upsamplers.0.conv");
                if (pos == std::string::npos) {
                    len = 44;
                    pos = tensor_storage.name.find("model.diffusion_model.output_blocks.2.1.conv");
                }
                if (pos != std::string::npos) {
                    tensor_storage.name = "model.diffusion_model.output_blocks.2.2.conv" + tensor_storage.name.substr(len);
                    LOG_DEBUG("NEW NAME: %s", tensor_storage.name.c_str());
                    add_preprocess_tensor_storage_types(tensor_storages_types, tensor_storage.name, tensor_storage.type);
                }
            }
            break;
        }
    }

    if (!init_from_safetensors_file(vae_path, "vae.")) {
        LOG_WARN("Couldn't find working VAE in %s", file_path.c_str());
        // return false;
    }
    if (!init_from_safetensors_file(clip_path, "te.")) {
        LOG_WARN("Couldn't find working text encoder in %s", file_path.c_str());
        // return false;
    }
    if (!init_from_safetensors_file(clip_g_path, "te.1.")) {
        LOG_DEBUG("Couldn't find working second text encoder in %s", file_path.c_str());
    }
    return true;
}

/*================================================= CkptModelLoader ==================================================*/

// $ python -m pickletools sd-v1-4/archive/data.pkl | head -n 100
//     0: \x80 PROTO      2
//     2: }    EMPTY_DICT
//     3: q    BINPUT     0
//     5: (    MARK
//     6: X        BINUNICODE 'epoch'
//    16: q        BINPUT     1
//    18: K        BININT1    6
//    20: X        BINUNICODE 'global_step'
//    36: q        BINPUT     2
//    38: J        BININT     470000
//    43: X        BINUNICODE 'pytorch-lightning_version'
//    73: q        BINPUT     3
//    75: X        BINUNICODE '1.4.2'
//    85: q        BINPUT     4
//    87: X        BINUNICODE 'state_dict'
//   102: q        BINPUT     5
//   104: }        EMPTY_DICT
//   105: q        BINPUT     6
//   107: (        MARK
//   108: X            BINUNICODE 'betas'
//   118: q            BINPUT     7
//   120: c            GLOBAL     'torch._utils _rebuild_tensor_v2'
//   153: q            BINPUT     8
//   155: (            MARK
//   156: (                MARK
//   157: X                    BINUNICODE 'storage'
//   169: q                    BINPUT     9
//   171: c                    GLOBAL     'torch FloatStorage'
//   191: q                    BINPUT     10
//   193: X                    BINUNICODE '0'
//   199: q                    BINPUT     11
//   201: X                    BINUNICODE 'cpu'
//   209: q                    BINPUT     12
//   211: M                    BININT2    1000
//   214: t                    TUPLE      (MARK at 156)
//   215: q                BINPUT     13
//   217: Q                BINPERSID
//   218: K                BININT1    0
//   220: M                BININT2    1000
//  ...............................
//  3201: q            BINPUT     250
//  3203: R            REDUCE
//  3204: q            BINPUT     251
//  3206: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.weight'
//  3264: q            BINPUT     252
//  3266: h            BINGET     8
//  3268: (            MARK
//  3269: (                MARK
//  3270: h                    BINGET     9
//  3272: h                    BINGET     10
//  3274: X                    BINUNICODE '30'
//  3281: q                    BINPUT     253
//  3283: h                    BINGET     12
//  3285: J                    BININT     102400
//  3290: t                    TUPLE      (MARK at 3269)
//  3291: q                BINPUT     254
//  3293: Q                BINPERSID
//  3294: K                BININT1    0
//  3296: (                MARK
//  3297: M                    BININT2    320
//  3300: M                    BININT2    320
//  3303: K                    BININT1    1
//  3305: K                    BININT1    1
//  3307: t                    TUPLE      (MARK at 3296)
//  3308: q                BINPUT     255
//  3310: (                MARK
//  3311: M                    BININT2    320
//  3314: K                    BININT1    1
//  3316: K                    BININT1    1
//  3318: K                    BININT1    1
//  3320: t                    TUPLE      (MARK at 3310)
//  3321: r                LONG_BINPUT 256
//  3326: \x89             NEWFALSE
//  3327: h                BINGET     16
//  3329: )                EMPTY_TUPLE
//  3330: R                REDUCE
//  3331: r                LONG_BINPUT 257
//  3336: t                TUPLE      (MARK at 3268)
//  3337: r            LONG_BINPUT 258
//  3342: R            REDUCE
//  3343: r            LONG_BINPUT 259
//  3348: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.bias'
//  3404: r            LONG_BINPUT 260
//  3409: h            BINGET     8
//  3411: (            MARK
//  3412: (                MARK
//  3413: h                    BINGET     9
//  3415: h                    BINGET     10
//  3417: X                    BINUNICODE '31'

struct PickleTensorReader {
    enum ReadPhase {
        READ_NAME,
        READ_DATA,
        CHECK_SIZE,
        READ_DIMENS
    };
    ReadPhase phase   = READ_NAME;
    size_t entry_size = 0;
    int32_t nelements = 0;

    TensorStorage tensor_storage;

    static ggml_type global_type;  // all pickle_tensors data type
    static bool read_global_type;

    bool read_int_value(uint32_t value) {
        if (phase == CHECK_SIZE) {
            if (entry_size == value * ggml_type_size(tensor_storage.type)) {
                nelements = value;
                phase     = READ_DIMENS;
                return true;
            } else {
                phase = READ_NAME;
            }
        } else if (phase == READ_DIMENS) {
            if (tensor_storage.n_dims + 1 > SD_MAX_DIMS) {  // too many dimens
                phase                 = READ_NAME;
                tensor_storage.n_dims = 0;
            }
            if (nelements % value == 0) {
                tensor_storage.ne[tensor_storage.n_dims] = value;
                tensor_storage.n_dims++;
            }
        }
        return false;
    }

    void read_global(const std::string& str) {
        if (str == "FloatStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F32;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F32;
        } else if (str == "HalfStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F16;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F16;
        }
    }

    void read_string(const std::string& str, struct zip_t* zip, std::string dir) {
        if (str == "storage") {
            read_global_type = true;
        } else if (str != "state_dict") {
            if (phase == READ_DATA) {
                std::string entry_name = dir + "data/" + std::string(str);

                size_t i, n = zip_entries_total(zip);
                for (i = 0; i < n; ++i) {
                    zip_entry_openbyindex(zip, i);
                    {
                        std::string name = zip_entry_name(zip);
                        if (name == entry_name) {
                            tensor_storage.index_in_zip = (int)i;
                            entry_size                  = zip_entry_size(zip);
                            zip_entry_close(zip);
                            break;
                        }
                    }
                    zip_entry_close(zip);
                }

                phase = entry_size > 0 ? CHECK_SIZE : READ_NAME;
            }
            if (!read_global_type && phase == READ_NAME) {
                tensor_storage.name = str;
                phase               = READ_DATA;
                tensor_storage.type = global_type;
            }
        }
    }
};

ggml_type PickleTensorReader::global_type = GGML_TYPE_F32;  // all pickle_tensors data type
bool PickleTensorReader::read_global_type = false;

int find_char(uint8_t* buffer, int len, char c) {
    for (int pos = 0; pos < len; pos++) {
        if (buffer[pos] == c) {
            return pos;
        }
    }
    return -1;
}

#define MAX_STRING_BUFFER 512

bool ModelLoader::parse_data_pkl(uint8_t* buffer,
                                 size_t buffer_size,
                                 zip_t* zip,
                                 std::string dir,
                                 size_t file_index,
                                 const std::string prefix) {
    uint8_t* buffer_end = buffer + buffer_size;
    if (buffer[0] == 0x80) {  // proto
        if (buffer[1] != 2) {
            LOG_ERROR("Unsupported protocol\n");
            return false;
        }
        buffer += 2;  // 0x80 and version
        char string_buffer[MAX_STRING_BUFFER];
        bool finish = false;
        PickleTensorReader reader;
        // read pickle binary file
        while (!finish && buffer < buffer_end) {
            uint8_t opcode = *buffer;
            buffer++;
            // https://github.com/python/cpython/blob/3.7/Lib/pickletools.py#L1048
            // https://github.com/python/cpython/blob/main/Lib/pickle.py#L105
            switch (opcode) {
                case '}':  // EMPTY_DICT     = b'}'   # push empty dict
                    break;
                case ']':  // EMPTY_LIST     = b']'   # push empty list
                    break;
                // skip unused sections
                case 'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
                case 'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
                case 'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
                    buffer++;
                    break;
                case 'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
                    buffer += 4;
                    break;
                case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                    buffer += 8;
                    break;
                case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
                    break;
                case '(':  // MARK           = b'('   # push special markobject on stack
                    break;
                case 'K':  // BININT1        = b'K'   # push 1-byte unsigned int
                {
                    uint8_t value = *buffer;
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer++;
                } break;
                case 'M':  // BININT2        = b'M'   # push 2-byte unsigned int
                {
                    uint16_t value = read_short(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer += 2;
                } break;
                case 'J':  // BININT         = b'J'   # push four-byte signed int
                {
                    const int32_t value = read_int(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;  // skip tuple after read num_elements
                    }
                    buffer += 4;
                } break;
                case 'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
                {
                    const int32_t len = read_int(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    if (len > MAX_STRING_BUFFER) {
                        LOG_WARN("tensor name very large");
                    }
                    memcpy(string_buffer, buffer, len < MAX_STRING_BUFFER ? len : (MAX_STRING_BUFFER - 1));
                    buffer += len;
                    reader.read_string(string_buffer, zip, dir);
                } break;
                case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
                {
                    const int8_t len = *buffer;
                    buffer++;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    // printf("String: '%s'\n", string_buffer);
                } break;
                case 'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
                {
                    int len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    buffer += len + 1;
                    len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len + 1;
                    reader.read_global(string_buffer);
                } break;
                case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
                case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
                case 't':   // TUPLE          = b't'   # build tuple from topmost stack items
                    if (reader.phase == PickleTensorReader::READ_DIMENS) {
                        reader.tensor_storage.reverse_ne();
                        reader.tensor_storage.file_index = file_index;
                        // if(strcmp(prefix.c_str(), "scarlett") == 0)
                        // printf(" ZIP got tensor %s \n ", reader.tensor_storage.name.c_str());
                        std::string name = reader.tensor_storage.name;
                        if (!starts_with(name, prefix)) {
                            name = prefix + name;
                        }
                        reader.tensor_storage.name = name;
                        tensor_storages.push_back(reader.tensor_storage);
                        add_preprocess_tensor_storage_types(tensor_storages_types, reader.tensor_storage.name, reader.tensor_storage.type);

                        // LOG_DEBUG("%s", reader.tensor_storage.name.c_str());
                        // reset
                        reader = PickleTensorReader();
                    }
                    break;
                case '.':  // STOP           = b'.'   # every pickle ends with STOP
                    finish = true;
                    break;
                default:
                    break;
            }
        }
    }
    return true;
}

bool ModelLoader::init_from_ckpt_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }
    int n = (int)zip_entries_total(zip);
    for (int i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        {
            std::string name = zip_entry_name(zip);
            size_t pos       = name.find("data.pkl");
            if (pos != std::string::npos) {
                std::string dir = name.substr(0, pos);
                printf("ZIP %d, name = %s, dir = %s \n", i, name.c_str(), dir.c_str());
                void* pkl_data = NULL;
                size_t pkl_size;
                zip_entry_read(zip, &pkl_data, &pkl_size);

                // LOG_DEBUG("%lld", pkl_size);

                parse_data_pkl((uint8_t*)pkl_data, pkl_size, zip, dir, file_index, prefix);

                free(pkl_data);
            }
        }
        zip_entry_close(zip);
    }
    zip_close(zip);
    return true;
}

bool ModelLoader::model_is_unet() {
    for (auto& tensor_storage : tensor_storages) {
        if (tensor_storage.name.find("model.diffusion_model.input_blocks.") != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool ModelLoader::has_diffusion_model_tensors()
{
    for (auto& tensor_storage : tensor_storages) {
        if (tensor_storage.name.find("model.diffusion_model.") != std::string::npos) {
            return true;
        }
    }
    return false;
}

SDVersion ModelLoader::get_sd_version() {
    TensorStorage token_embedding_weight, input_block_weight;
    bool input_block_checked = false;

    bool has_multiple_encoders = false;
    bool is_unet               = false;

    bool is_xl                       = false;
    bool is_flux                     = false;
    bool is_wan                      = false;
    int64_t patch_embedding_channels = 0;
    bool has_img_emb                 = false;

    for (auto& tensor_storage : tensor_storages) {
        if (!(is_xl || is_flux)) {
            if (tensor_storage.name.find("model.diffusion_model.double_blocks.") != std::string::npos) {
                is_flux = true;
                if (input_block_checked) {
                    break;
                }
            }
            if (tensor_storage.name.find("model.diffusion_model.joint_blocks.") != std::string::npos) {
                return VERSION_SD3;
            }
            if (tensor_storage.name.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight") != std::string::npos) {
                is_wan = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.patch_embedding.weight") != std::string::npos) {
                patch_embedding_channels = tensor_storage.ne[3];
            }
            if (tensor_storage.name.find("model.diffusion_model.img_emb") != std::string::npos) {
                has_img_emb = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.input_blocks.") != std::string::npos || tensor_storage.name.find("unet.down_blocks.") != std::string::npos) {
                is_unet = true;
                if (has_multiple_encoders) {
                    is_xl = true;
                    if (input_block_checked) {
                        break;
                    }
                }
            }
            if (tensor_storage.name.find("conditioner.embedders.1") != std::string::npos || tensor_storage.name.find("cond_stage_model.1") != std::string::npos || tensor_storage.name.find("te.1") != std::string::npos) {
                has_multiple_encoders = true;
                if (is_unet) {
                    is_xl = true;
                    if (input_block_checked) {
                        break;
                    }
                }
            }
            if (tensor_storage.name.find("model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor") != std::string::npos) {
                return VERSION_SVD;
            }
        }
        if (tensor_storage.name == "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "cond_stage_model.model.token_embedding.weight" ||
            tensor_storage.name == "text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "te.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.model.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") {
            token_embedding_weight = tensor_storage;
            // break;
        }
        if (tensor_storage.name == "model.diffusion_model.input_blocks.0.0.weight" || tensor_storage.name == "model.diffusion_model.img_in.weight" || tensor_storage.name == "unet.conv_in.weight") {
            input_block_weight  = tensor_storage;
            input_block_checked = true;
            if (is_xl || is_flux) {
                break;
            }
        }
    }
    if (is_wan) {
        LOG_DEBUG("patch_embedding_channels %d", patch_embedding_channels);
        if (patch_embedding_channels == 184320 && !has_img_emb) {
            return VERSION_WAN2_2_I2V;
        }
        if (patch_embedding_channels == 147456 && !has_img_emb) {
            return VERSION_WAN2_2_TI2V;
        }
        return VERSION_WAN2;
    }
    bool is_inpaint = input_block_weight.ne[2] == 9;
    bool is_ip2p    = input_block_weight.ne[2] == 8;
    if (is_xl) {
        if (is_inpaint) {
            return VERSION_SDXL_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SDXL_PIX2PIX;
        }
        return VERSION_SDXL;
    }

    if (is_flux) {
        is_inpaint = input_block_weight.ne[0] == 384;
        if (is_inpaint) {
            return VERSION_FLUX_FILL;
        }
        return VERSION_FLUX;
    }

    if (token_embedding_weight.ne[0] == 768) {
        if (is_inpaint) {
            return VERSION_SD1_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SD1_PIX2PIX;
        }
        return VERSION_SD1;
    } else if (token_embedding_weight.ne[0] == 1024) {
        if (is_inpaint) {
            return VERSION_SD2_INPAINT;
        }
        return VERSION_SD2;
    }
    return VERSION_COUNT;
}

ggml_type ModelLoader::get_sd_wtype() {
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (ggml_is_quantized(tensor_storage.type)) {
            return tensor_storage.type;
        }

        if (tensor_should_be_converted(tensor_storage, GGML_TYPE_Q4_K)) {
            return tensor_storage.type;
        }
    }
    return GGML_TYPE_COUNT;
}

ggml_type ModelLoader::get_conditioner_wtype() {
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if ((tensor_storage.name.find("text_encoders") == std::string::npos &&
             tensor_storage.name.find("cond_stage_model") == std::string::npos &&
             tensor_storage.name.find("te.text_model.") == std::string::npos &&
             tensor_storage.name.find("conditioner") == std::string::npos)) {
            continue;
        }

        if (ggml_is_quantized(tensor_storage.type)) {
            return tensor_storage.type;
        }

        if (tensor_should_be_converted(tensor_storage, GGML_TYPE_Q4_K)) {
            return tensor_storage.type;
        }
    }
    return GGML_TYPE_COUNT;
}

ggml_type ModelLoader::get_diffusion_model_wtype() {
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("model.diffusion_model.") == std::string::npos && tensor_storage.name.find("unet.") == std::string::npos) {
            continue;
        }

        if (ggml_is_quantized(tensor_storage.type)) {
            return tensor_storage.type;
        }

        if (tensor_should_be_converted(tensor_storage, GGML_TYPE_Q4_K)) {
            return tensor_storage.type;
        }
    }
    return GGML_TYPE_COUNT;
}

ggml_type ModelLoader::get_vae_wtype() {
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("vae.") == std::string::npos &&
            tensor_storage.name.find("first_stage_model") == std::string::npos) {
            continue;
        }

        if (ggml_is_quantized(tensor_storage.type)) {
            return tensor_storage.type;
        }

        if (tensor_should_be_converted(tensor_storage, GGML_TYPE_Q4_K)) {
            return tensor_storage.type;
        }
    }
    return GGML_TYPE_COUNT;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string prefix) {
    for (auto& pair : tensor_storages_types) {
        if (prefix.size() < 1 || pair.first.substr(0, prefix.size()) == prefix) {
            bool found = false;
            for (auto& tensor_storage : tensor_storages) {
                std::map<std::string, ggml_type> temp;
                add_preprocess_tensor_storage_types(temp, tensor_storage.name, tensor_storage.type);
                for (auto& preprocessed_name : temp) {
                    if (preprocessed_name.first == pair.first) {
                        if (tensor_should_be_converted(tensor_storage, wtype)) {
                            pair.second = wtype;
                        }
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
        }
    }
}

std::string ModelLoader::load_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(merges_utf8_c_str), sizeof(merges_utf8_c_str));
    return merges_utf8_str;
}

std::string ModelLoader::load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string ModelLoader::load_umt5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
    return json_str;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads_p) {
    int64_t process_time_ms = 0;
    std::atomic<int64_t> read_time_ms(0);
    std::atomic<int64_t> memcpy_time_ms(0);
    std::atomic<int64_t> copy_to_backend_time_ms(0);
    std::atomic<int64_t> convert_time_ms(0);

    int num_threads_to_use = n_threads_p > 0 ? n_threads_p : sd_get_num_physical_cores();
    LOG_DEBUG("using %d threads for model loading", num_threads_to_use);

    int64_t start_time = ggml_time_ms();
    std::vector<TensorStorage> processed_tensor_storages;

    {
        struct IndexedStorage {
            size_t index;
            TensorStorage ts;
        };

        std::mutex vec_mutex;
        std::vector<IndexedStorage> all_results;

        int n_threads = std::min(num_threads_to_use, (int)tensor_storages.size());
        if (n_threads < 1) {
            n_threads = 1;
        }
        std::vector<std::thread> workers;

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, thread_id = i]() {
                std::vector<IndexedStorage> local_results;
                std::vector<TensorStorage> temp_storages;

                for (size_t j = thread_id; j < tensor_storages.size(); j += n_threads) {
                    const auto& tensor_storage = tensor_storages[j];
                    if (is_unused_tensor(tensor_storage.name)) {
                        continue;
                    }

                    temp_storages.clear();
                    preprocess_tensor(tensor_storage, temp_storages);

                    for (const auto& ts : temp_storages) {
                        local_results.push_back({j, ts});
                    }
                }

                if (!local_results.empty()) {
                    std::lock_guard<std::mutex> lock(vec_mutex);
                    all_results.insert(all_results.end(),
                                       local_results.begin(), local_results.end());
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }

        std::vector<IndexedStorage> deduplicated;
        deduplicated.reserve(all_results.size());
        std::unordered_map<std::string, size_t> name_to_pos;
        for (auto& entry : all_results) {
            auto it = name_to_pos.find(entry.ts.name);
            if (it == name_to_pos.end()) {
                name_to_pos.emplace(entry.ts.name, deduplicated.size());
                deduplicated.push_back(entry);
            } else if (deduplicated[it->second].index < entry.index) {
                deduplicated[it->second] = entry;
            }
        }

        std::sort(deduplicated.begin(), deduplicated.end(), [](const IndexedStorage& a, const IndexedStorage& b) {
            return a.index < b.index;
        });

        processed_tensor_storages.reserve(deduplicated.size());
        for (auto& entry : deduplicated) {
            processed_tensor_storages.push_back(entry.ts);
        }
    }

    process_time_ms = ggml_time_ms() - start_time;

    bool success                          = true;
    size_t total_tensors_processed        = 0;
    const size_t total_tensors_to_process = processed_tensor_storages.size();
    const int64_t t_start                 = ggml_time_ms();
    int last_n_threads                    = 1;

    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];
        LOG_DEBUG("loading tensors from %s", file_path.c_str());

        std::vector<const TensorStorage*> file_tensors;
        for (const auto& ts : processed_tensor_storages) {
            if (ts.file_index == file_index) {
                file_tensors.push_back(&ts);
            }
        }
        if (file_tensors.empty()) {
            continue;
        }

        bool is_zip = false;
        for (auto const& ts : file_tensors) {
            if (ts->index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        int n_threads = is_zip ? 1 : std::min(num_threads_to_use, (int)file_tensors.size());
        if (n_threads < 1) {
            n_threads = 1;
        }
        last_n_threads = n_threads;

        std::atomic<size_t> tensor_idx(0);
        std::atomic<bool> failed(false);
        std::vector<std::thread> workers;

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, file_path, is_zip]() {
                std::ifstream file;
                struct zip_t* zip = NULL;
                if (is_zip) {
                    zip = zip_open(file_path.c_str(), 0, 'r');
                    if (zip == NULL) {
                        LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                } else {
                    // kcpp
                    #ifdef _WIN32
                        std::filesystem::path fpath = std::filesystem::u8path(file_path);
                    #else
                        std::filesystem::path fpath = std::filesystem::path(file_path);
                    #endif
                    file.open(fpath, std::ios::binary);
                    if (!file.is_open()) {
                        LOG_ERROR("failed to open '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                }

                std::vector<uint8_t> read_buffer;
                std::vector<uint8_t> convert_buffer;

                while (true) {
                    int64_t t0, t1;
                    size_t idx = tensor_idx.fetch_add(1);
                    if (idx >= file_tensors.size() || failed) {
                        break;
                    }

                    const TensorStorage& tensor_storage = *file_tensors[idx];
                    ggml_tensor* dst_tensor             = NULL;

                    t0 = ggml_time_ms();

                    if (!on_new_tensor_cb(tensor_storage, &dst_tensor)) {
                        LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                        failed = true;
                        break;
                    }

                    if (dst_tensor == NULL) {
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);
                        continue;
                    }

                    size_t nbytes_to_read = tensor_storage.nbytes_to_read();

                    auto read_data = [&](char* buf, size_t n) {
                        if (zip != NULL) {
                            zip_entry_openbyindex(zip, tensor_storage.index_in_zip);
                            size_t entry_size = zip_entry_size(zip);
                            if (entry_size != n) {
                                int64_t t_memcpy_start;
                                read_buffer.resize(entry_size);
                                zip_entry_noallocread(zip, (void*)read_buffer.data(), entry_size);
                                t_memcpy_start = ggml_time_ms();
                                memcpy((void*)buf, (void*)(read_buffer.data() + tensor_storage.offset), n);
                                memcpy_time_ms.fetch_add(ggml_time_ms() - t_memcpy_start);
                            } else {
                                zip_entry_noallocread(zip, (void*)buf, n);
                            }
                            zip_entry_close(zip);
                        } else {
                            file.seekg(tensor_storage.offset);
                            file.read(buf, n);
                            if (!file) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                failed = true;
                            }
                        }
                    };

                    if (dst_tensor->buffer == NULL || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        if (tensor_storage.type == dst_tensor->type) {
                            GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                            if (tensor_storage.is_f64 || tensor_storage.is_i64) {
                                read_buffer.resize(tensor_storage.nbytes_to_read());
                                read_data((char*)read_buffer.data(), nbytes_to_read);
                            } else {
                                read_data((char*)dst_tensor->data, nbytes_to_read);
                            }
                            t1 = ggml_time_ms();
                            read_time_ms.fetch_add(t1 - t0);

                            t0 = ggml_time_ms();
                            if (tensor_storage.is_bf16) {
                                // inplace op
                                bf16_to_f32_vec((uint16_t*)dst_tensor->data, (float*)dst_tensor->data, tensor_storage.nelements());
                            } else if (tensor_storage.is_f8_e4m3) {
                                // inplace op
                                f8_e4m3_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                            } else if (tensor_storage.is_f8_e5m2) {
                                // inplace op
                                f8_e5m2_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                            } else if (tensor_storage.is_f64) {
                                f64_to_f32_vec((double*)read_buffer.data(), (float*)dst_tensor->data, tensor_storage.nelements());
                            } else if (tensor_storage.is_i64) {
                                i64_to_i32_vec((int64_t*)read_buffer.data(), (int32_t*)dst_tensor->data, tensor_storage.nelements());
                            }
                            t1 = ggml_time_ms();
                            convert_time_ms.fetch_add(t1 - t0);
                        } else {
                            read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                            read_data((char*)read_buffer.data(), nbytes_to_read);
                            t1 = ggml_time_ms();
                            read_time_ms.fetch_add(t1 - t0);

                            t0 = ggml_time_ms();
                            if (tensor_storage.is_bf16) {
                                // inplace op
                                bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                            } else if (tensor_storage.is_f8_e4m3) {
                                // inplace op
                                f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                            } else if (tensor_storage.is_f8_e5m2) {
                                // inplace op
                                f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                            } else if (tensor_storage.is_f64) {
                                // inplace op
                                f64_to_f32_vec((double*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                            } else if (tensor_storage.is_i64) {
                                // inplace op
                                i64_to_i32_vec((int64_t*)read_buffer.data(), (int32_t*)read_buffer.data(), tensor_storage.nelements());
                            }
                            convert_tensor((void*)read_buffer.data(), tensor_storage.type, dst_tensor->data, dst_tensor->type, (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                            t1 = ggml_time_ms();
                            convert_time_ms.fetch_add(t1 - t0);
                        }
                    } else {
                        read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                        read_data((char*)read_buffer.data(), nbytes_to_read);
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);

                        t0 = ggml_time_ms();
                        if (tensor_storage.is_bf16) {
                            // inplace op
                            bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                        } else if (tensor_storage.is_f8_e4m3) {
                            // inplace op
                            f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                        } else if (tensor_storage.is_f8_e5m2) {
                            // inplace op
                            f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                        } else if (tensor_storage.is_f64) {
                            // inplace op
                            f64_to_f32_vec((double*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                        } else if (tensor_storage.is_i64) {
                            // inplace op
                            i64_to_i32_vec((int64_t*)read_buffer.data(), (int32_t*)read_buffer.data(), tensor_storage.nelements());
                        }

                        if (tensor_storage.type == dst_tensor->type) {
                            // copy to device memory
                            t1 = ggml_time_ms();
                            convert_time_ms.fetch_add(t1 - t0);
                            t0 = ggml_time_ms();
                            ggml_backend_tensor_set(dst_tensor, read_buffer.data(), 0, ggml_nbytes(dst_tensor));
                            t1 = ggml_time_ms();
                            copy_to_backend_time_ms.fetch_add(t1 - t0);
                        } else {
                            // convert first, then copy to device memory

                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_tensor((void*)read_buffer.data(), tensor_storage.type, (void*)convert_buffer.data(), dst_tensor->type, (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                            t1 = ggml_time_ms();
                            convert_time_ms.fetch_add(t1 - t0);
                            t0 = ggml_time_ms();
                            ggml_backend_tensor_set(dst_tensor, convert_buffer.data(), 0, ggml_nbytes(dst_tensor));
                            t1 = ggml_time_ms();
                            copy_to_backend_time_ms.fetch_add(t1 - t0);
                        }
                    }
                }
                if (zip != NULL) {
                    zip_close(zip);
                }
            });
        }

        while (true) {
            size_t current_idx = tensor_idx.load();
            if (current_idx >= file_tensors.size() || failed) {
                break;
            }
            size_t curr_num = total_tensors_processed + current_idx;
            pretty_progress(curr_num, total_tensors_to_process, (ggml_time_ms() - t_start) / 1000.0f / (curr_num + 1e-6f));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        for (auto& w : workers) {
            w.join();
        }

        if (failed) {
            success = false;
            break;
        }
        total_tensors_processed += file_tensors.size();
        pretty_progress(total_tensors_processed, total_tensors_to_process, (ggml_time_ms() - t_start) / 1000.0f / (total_tensors_processed + 1e-6f));
        if (total_tensors_processed < total_tensors_to_process) {
            printf("\n");
        }
    }

    int64_t end_time = ggml_time_ms();
    LOG_INFO("loading tensors completed, taking %.2fs (process: %.2fs, read: %.2fs, memcpy: %.2fs, convert: %.2fs, copy_to_backend: %.2fs)",
             (end_time - start_time) / 1000.f,
             process_time_ms / 1000.f,
             (read_time_ms.load() / (float)last_n_threads) / 1000.f,
             (memcpy_time_ms.load() / (float)last_n_threads) / 1000.f,
             (convert_time_ms.load() / (float)last_n_threads) / 1000.f,
             (copy_to_backend_time_ms.load() / (float)last_n_threads) / 1000.f);
    return success;
}

bool ModelLoader::load_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                               std::set<std::string> ignore_tensors,
                               int n_threads) {
    std::set<std::string> tensor_names_in_file;
    std::mutex tensor_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
        {
            std::lock_guard<std::mutex> lock(tensor_names_mutex);
            tensor_names_in_file.insert(name);
        }

        struct ggml_tensor* real;
        if (tensors.find(name) != tensors.end()) {
            real = tensors[name];
        } else {
            for (auto& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        if (
            real->ne[0] != tensor_storage.ne[0] ||
            real->ne[1] != tensor_storage.ne[1] ||
            real->ne[2] != tensor_storage.ne[2] ||
            real->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "tensor '%s' has wrong shape in model file: "
                "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
            return false;
        }

        *dst_tensor = real;

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, n_threads);
    if (!success) {
        LOG_ERROR("load tensors from file failed");
        return false;
    }

    bool some_tensor_not_init = false;

    for (auto pair : tensors) {
        if (pair.first.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) {
            continue;
        }

        if (pair.first.find("alphas_cumprod") != std::string::npos) {
            continue;
        }

        if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
            some_tensor_not_init = true;
        }
    }

    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

std::vector<std::pair<std::string, ggml_type>> parse_tensor_type_rules(const std::string& tensor_type_rules) {
    std::vector<std::pair<std::string, ggml_type>> result;
    for (const auto& item : split_string(tensor_type_rules, ',')) {
        if (item.size() == 0)
            continue;
        std::string::size_type pos = item.find('=');
        if (pos == std::string::npos) {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
            continue;
        }
        std::string tensor_pattern = item.substr(0, pos);
        std::string type_name      = item.substr(pos + 1);

        ggml_type tensor_type = GGML_TYPE_COUNT;

        if (type_name == "f32") {
            tensor_type = GGML_TYPE_F32;
        } else {
            for (size_t i = 0; i < GGML_TYPE_COUNT; i++) {
                auto trait = ggml_get_type_traits((ggml_type)i);
                if (trait->to_float && trait->type_size && type_name == trait->type_name) {
                    tensor_type = (ggml_type)i;
                }
            }
        }

        if (tensor_type != GGML_TYPE_COUNT) {
            result.emplace_back(tensor_pattern, tensor_type);
        } else {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
        }
    }
    return result;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
    const std::string& name = tensor_storage.name;
    if (type != GGML_TYPE_COUNT) {
        if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
            // Pass, do not convert
        } else if (ends_with(name, ".bias")) {
            // Pass, do not convert
        } else if (ends_with(name, ".scale")) {
            // Pass, do not convert
        } else if (contains(name, "img_in.") ||
                   contains(name, "txt_in.") ||
                   contains(name, "time_in.") ||
                   contains(name, "vector_in.") ||
                   contains(name, "guidance_in.") ||
                   contains(name, "final_layer.")) {
            // Pass, do not convert. For FLUX
        } else if (contains(name, "x_embedder.") ||
                   contains(name, "t_embedder.") ||
                   contains(name, "y_embedder.") ||
                   contains(name, "pos_embed") ||
                   contains(name, "context_embedder.")) {
            // Pass, do not convert. For MMDiT
        } else if (contains(name, "time_embed.") || contains(name, "label_emb.")) {
            // Pass, do not convert. For Unet
        } else if (contains(name, "embedding")) {
            // Pass, do not convert embedding
        } else {
            return true;
        }
    }
    return false;
}

bool ModelLoader::save_to_gguf_file(const std::string& file_path, ggml_type type, const std::string& tensor_type_rules_str) {
    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += tensor_storages.size() * ggml_tensor_overhead();
    mem_size += get_params_mem_size(backend, type);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, NULL, false});

    gguf_context* gguf_ctx = gguf_init_empty();

    auto tensor_type_rules = parse_tensor_type_rules(tensor_type_rules_str);

    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = tensor_storage.type;
        ggml_type dst_type      = type;

        for (const auto& tensor_type_rule : tensor_type_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }

        if (tensor_should_be_converted(tensor_storage, dst_type)) {
            tensor_type = dst_type;
        }

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == NULL) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        // LOG_DEBUG("%s %d %s %d[%d %d %d %d] %d[%d %d %d %d]", name.c_str(),
        // ggml_nbytes(tensor), ggml_type_name(tensor_type),
        // tensor_storage.n_dims,
        // tensor_storage.ne[0], tensor_storage.ne[1], tensor_storage.ne[2], tensor_storage.ne[3],
        // tensor->n_dims, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

        *dst_tensor = tensor;

        gguf_add_tensor(gguf_ctx, tensor);

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb);
    ggml_backend_free(backend);
    LOG_INFO("load tensors done");
    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    if (success) {
        gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    }
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
    return success;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) {
    size_t alignment = 128;
    if (backend != NULL) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    std::vector<TensorStorage> processed_tensor_storages;
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        preprocess_tensor(tensor_storage, processed_tensor_storages);
    }

    for (auto& tensor_storage : processed_tensor_storages) {
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}

bool convert(const char* input_path, const char* vae_path, const char* output_path, sd_type_t output_type, const char* tensor_type_rules) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != NULL && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    bool success = model_loader.save_to_gguf_file(output_path, (ggml_type)output_type, tensor_type_rules);
    return success;
}

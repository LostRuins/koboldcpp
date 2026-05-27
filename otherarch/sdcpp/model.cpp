#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/torch_legacy_io.h"
#include "model_io/torch_zip_io.h"
#include "stable-diffusion.h"
#include "util.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml_extend_backend.h"
#include "zip.h"

#include "name_conversion.h"

/*================================================= Preprocess ==================================================*/

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
    "cond_stage_model.1.model.text_model.embeddings.position_ids",
    "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
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
    "ztsnr",                                                        // Found in some SDXL vpred models
    "edm_vpred.sigma_min",                                          // Found in CosXL
    // TODO: find another way to avoid the "unknown tensor" for these two
    // "edm_vpred.sigma_max", // Used to detect CosXL
    // "v_pred", // Used to detect SDXL vpred models
    "text_encoders.llm.output.weight",
    "text_encoders.llm.lm_head.",
    "first_stage_model.bn.",
};

bool is_unused_tensor(const std::string& name) {
    for (size_t i = 0; i < sizeof(unused_tensors) / sizeof(const char*); i++) {
        if (starts_with(name, unused_tensors[i])) {
            return true;
        }
    }
    return false;
}

std::string kcpp_fix_wrong_img_tensor_name(const std::string& name) //kcpp function that fixes common wrong tensor names
{
    if (starts_with(name, "text_encoders.qwen25_7b.transformer.model.")) {
        return "text_encoders.llm.model." + name.substr(strlen("text_encoders.qwen25_7b.transformer.model."));
    }
    if (starts_with(name, "text_encoders.qwen25_7b.transformer.visual.")) {
        return "text_encoders.llm.visual." + name.substr(strlen("text_encoders.qwen25_7b.transformer.visual."));
    }
    if (starts_with(name, "text_encoders.umt5xxl.")) {
        return "text_encoders.t5xxl." + name.substr(strlen("text_encoders.umt5xxl."));
    }
    return name;
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
    return static_cast<uint16_t>(fp8) << 8;
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
            if (qtype->to_float == nullptr) {
                throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
                                                   ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == nullptr) {
            throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
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

void ModelLoader::add_tensor_storage(const TensorStorage& tensor_storage) {
    tensor_storage_map[tensor_storage.name] = tensor_storage;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    return [&](const std::string& file_path) { // kcpp u8 file path
    if (is_directory(file_path)) {
        LOG_INFO("load %s using diffusers format", file_path.c_str());
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        LOG_INFO("load %s using safetensors format", file_path.c_str());
        return init_from_safetensors_file(file_path, prefix);
    } else if (is_torch_zip_file(file_path)) {
        LOG_INFO("load %s using torch zip format", file_path.c_str());
        return init_from_torch_zip_file(file_path, prefix);
    } else if (init_from_torch_legacy_file(file_path, prefix)) {
        LOG_INFO("load %s using torch legacy format", file_path.c_str());
        return true;
    } else {
        if (file_exists(file_path)) {
            LOG_WARN("unknown format %s", file_path.c_str());
        } else {
            LOG_WARN("file %s not found", file_path.c_str());
        }
        return false;
    }
    }(sd_get_u8path(file_path)); // kcpp u8 file path
}

void ModelLoader::convert_tensors_name() {
    SDVersion version = (version_ == VERSION_COUNT) ? get_sd_version() : version_;
    String2TensorStorage new_map;

    for (auto& [_, tensor_storage] : tensor_storage_map) {
        auto new_name = convert_tensor_name(tensor_storage.name, version);
        // LOG_DEBUG("%s -> %s", tensor_storage.name.c_str(), new_name.c_str());
        tensor_storage.name = new_name;
        new_map[new_name]   = std::move(tensor_storage);
    }

    tensor_storage_map.swap(new_map);
}

bool ModelLoader::init_from_file_and_convert_name(const std::string& file_path, const std::string& prefix, SDVersion version) {
    if (version_ == VERSION_COUNT && version != VERSION_COUNT) {
        version_ = version;
    }
    if (!init_from_file(file_path, prefix)) {
        return false;
    }
    convert_tensors_name();
    return true;
}

/*================================================= GGUFModelLoader ==================================================*/

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_gguf_file(file_path, tensor_storages, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    for (auto& tensor_storage : tensor_storages) {
        // LOG_DEBUG("%s", tensor_storage.name.c_str());

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);
    }

    return true;
}

/*================================================= SafeTensorsModelLoader ==================================================*/

bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_safetensors_file(file_path, tensor_storages, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        tensor_storage.name = kcpp_fix_wrong_img_tensor_name(tensor_storage.name);

        add_tensor_storage(tensor_storage);

        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
    }

    return true;
}

/*================================================= TorchLegacyModelLoader ==================================================*/

bool ModelLoader::init_from_torch_legacy_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from torch legacy '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_torch_legacy_file(file_path, tensor_storages, &error)) {
        if ((!error.empty()) && (ends_with(file_path, ".pt") || ends_with(file_path, ".pth"))) {
            LOG_WARN("%s", error.c_str());
        }
        return false;
    }

    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);
    }

    return true;
}

/*================================================= TorchZipModelLoader ==================================================*/

bool ModelLoader::init_from_torch_zip_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());

    std::vector<TensorStorage> tensor_storages;
    std::string error;
    if (!read_torch_zip_file(file_path, tensor_storages, &error)) {
        LOG_ERROR("%s", error.c_str());
        return false;
    }

    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    for (auto& tensor_storage : tensor_storages) {
        if (!starts_with(tensor_storage.name, prefix)) {
            tensor_storage.name = prefix + tensor_storage.name;
        }
        tensor_storage.file_index = file_index;

        add_tensor_storage(tensor_storage);

        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
    }

    return true;
}

bool ModelLoader::has_diffusion_model_tensors()
{
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (tensor_storage.name.find("model.diffusion_model.") != std::string::npos) {
            return true;
        }
    }
    return false;
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

SDVersion ModelLoader::get_sd_version() {
    TensorStorage token_embedding_weight, input_block_weight, context_ebedding_weight;

    bool has_multiple_encoders = false;
    bool is_unet               = false;

    bool is_xl                       = false;
    bool is_flux                     = false;
    bool is_flux2                    = false;
    bool has_single_block_47         = false;
    bool is_wan                      = false;
    int64_t patch_embedding_channels = 0;
    bool has_img_emb                 = false;
    bool has_middle_block_1          = false;
    bool has_output_block_311        = false;
    bool has_output_block_71         = false;
    bool has_attn_1024               = false;

    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (tensor_storage.name.find("model.diffusion_model.double_blocks.") != std::string::npos ||
            tensor_storage.name.find("model.diffusion_model.single_transformer_blocks.") != std::string::npos) {
            is_flux = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.nerf_final_layer_conv.") != std::string::npos) {
            return VERSION_CHROMA_RADIANCE;
        }
        if (tensor_storage.name.find("model.diffusion_model.joint_blocks.") != std::string::npos) {
            return VERSION_SD3;
        }
        if (tensor_storage.name.find("model.x_embedder.proj1.weight") != std::string::npos &&
            tensor_storage_map.find("model.language_model.layers.0.self_attn.q_proj.weight") != tensor_storage_map.end()) {
            return VERSION_HIDREAM_O1;
        }
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight") != std::string::npos &&
            tensor_storage_map.find("model.diffusion_model.transformer_blocks.0.img_mlp.w1.weight") != tensor_storage_map.end()) {
            return VERSION_LENS;
        }
        if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.img_mod.1.weight") != std::string::npos) {
            return VERSION_QWEN_IMAGE;
        }
        if (tensor_storage.name.find("llm_adapter.blocks.0.cross_attn.q_proj.weight") != std::string::npos) {
            return VERSION_ANIMA;
        }
        if (tensor_storage.name.find("model.diffusion_model.double_stream_modulation_img.lin.weight") != std::string::npos) {
            is_flux2 = true;
        }
        if (tensor_storage.name.find("single_blocks.47.linear1.weight") != std::string::npos) {
            has_single_block_47 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.double_blocks.0.img_mlp.gate_proj.weight") != std::string::npos) {
            return VERSION_OVIS_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.cap_embedder.0.weight") != std::string::npos) {
            return VERSION_Z_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.layers.0.adaLN_sa_ln.weight") != std::string::npos) {
            return VERSION_ERNIE_IMAGE;
        }
        if (tensor_storage.name.find("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias") != std::string::npos) {
            return VERSION_LTXAV;
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
        if (tensor_storage.name.find("model.diffusion_model.input_blocks.") != std::string::npos ||
            tensor_storage.name.find("unet.down_blocks.") != std::string::npos) {
            is_unet = true;
            if (has_multiple_encoders) {
                is_xl = true;
            }
        }
        if (tensor_storage.name.find("conditioner.embedders.1") != std::string::npos ||
            tensor_storage.name.find("cond_stage_model.1") != std::string::npos ||
            tensor_storage.name.find("te.1") != std::string::npos) {
            has_multiple_encoders = true;
            if (is_unet) {
                is_xl = true;
            }
        }
        if (tensor_storage.name.find("model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor") != std::string::npos) {
            return VERSION_SVD;
        }
        if (tensor_storage.name.find("model.diffusion_model.middle_block.1.") != std::string::npos ||
            tensor_storage.name.find("unet.mid_block.resnets.1.") != std::string::npos) {
            has_middle_block_1 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.3.1.transformer_blocks.1") != std::string::npos ||
            tensor_storage.name.find("unet.up_blocks.1.attentions.0.transformer_blocks.1") != std::string::npos) {
            has_output_block_311 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.7.1") != std::string::npos ||
            tensor_storage.name.find("unet.up_blocks.2.attentions.1") != std::string::npos) {
            has_output_block_71 = true;
            if (tensor_storage.name.find("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight") != std::string::npos) {
                if (tensor_storage.ne[0] == 1024)
                    has_attn_1024 = true;
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
        if (tensor_storage.name == "model.diffusion_model.input_blocks.0.0.weight" ||
            tensor_storage.name == "model.diffusion_model.img_in.weight" ||
            tensor_storage.name == "unet.conv_in.weight") {
            input_block_weight = tensor_storage;
        }
        if (tensor_storage.name == "model.diffusion_model.txt_in.weight" || tensor_storage.name == "model.diffusion_model.context_embedder.weight") {
            context_ebedding_weight = tensor_storage;
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
        if (!has_middle_block_1) {
            if (!has_output_block_311) {
                return VERSION_SDXL_VEGA;
            }
            return VERSION_SDXL_SSD1B;
        }
        return VERSION_SDXL;
    }

    if (is_flux && !is_flux2) {
        if (context_ebedding_weight.ne[0] == 3584) {
            return VERSION_LONGCAT;
        } else {
            if (input_block_weight.ne[0] == 384) {
                return VERSION_FLUX_FILL;
            }
            if (input_block_weight.ne[0] == 128) {
                return VERSION_FLUX_CONTROLS;
            }
            if (input_block_weight.ne[0] == 196) {
                return VERSION_FLEX_2;
            }
            return VERSION_FLUX;
        }
    }

    if (is_flux2) {
        if (has_single_block_47) {
            return VERSION_FLUX2;
        }
        return VERSION_FLUX2_KLEIN;
    }

    if (token_embedding_weight.ne[0] == 768) {
        if (is_inpaint) {
            return VERSION_SD1_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SD1_PIX2PIX;
        }
        if (!has_middle_block_1) {
            if (!has_output_block_71) {
                return VERSION_SDXS_512_DS;
            }
            return VERSION_SD1_TINY_UNET;
        }
        return VERSION_SD1;
    } else if (token_embedding_weight.ne[0] == 1024) {
        if (is_inpaint) {
            return VERSION_SD2_INPAINT;
        }
        if (!has_middle_block_1) {
            return has_attn_1024 ? VERSION_SDXS_09 : VERSION_SD2_TINY_UNET;
        }
        return VERSION_SD2;
    }
    return VERSION_COUNT;
}

std::map<ggml_type, uint32_t> ModelLoader::get_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_conditioner_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if ((tensor_storage.name.find("text_encoders") == std::string::npos &&
             tensor_storage.name.find("cond_stage_model") == std::string::npos &&
             tensor_storage.name.find("te.text_model.") == std::string::npos &&
             tensor_storage.name.find("conditioner") == std::string::npos)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_diffusion_model_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("model.diffusion_model.") == std::string::npos && tensor_storage.name.find("unet.") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_vae_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("vae.") == std::string::npos &&
            tensor_storage.name.find("first_stage_model") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

TensorTypeRules parse_tensor_type_rules(const std::string& tensor_type_rules) {
    TensorTypeRules result;
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

void ModelLoader::set_wtype_override(ggml_type wtype, std::string tensor_type_rules) {
    auto map_rules = parse_tensor_type_rules(tensor_type_rules);
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        ggml_type dst_type = wtype;
        for (const auto& tensor_type_rule : map_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }
        if (dst_type == GGML_TYPE_COUNT) {
            continue;
        }
        if (!tensor_should_be_converted(tensor_storage, dst_type)) {
            continue;
        }
        tensor_storage.expected_type = dst_type;
    }
}

void ModelLoader::process_model_files(bool enable_mmap, bool writable_mmap) {
    if (model_files_processed) {
        return;
    }

    int64_t start_time = ggml_time_ms();

    std::vector<TensorStorage> processed_tensor_storages;
    for (const auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        processed_tensor_storages.push_back(tensor_storage);
    }

    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];

        std::vector<TensorStorage> file_tensors;
        for (const auto& ts : processed_tensor_storages) {
            if (ts.file_index == file_index) {
                file_tensors.push_back(ts);
            }
        }
        if (file_tensors.empty()) {
            continue;
        }

        bool is_zip = false;
        for (auto const& ts : file_tensors) {
            if (ts.index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        ModelFileData fdata = {};
        fdata.path          = file_path;
        fdata.is_zip        = is_zip;
        fdata.tensors       = std::move(file_tensors);

        if (enable_mmap && !is_zip) {
            LOG_DEBUG("using mmap for I/O");
            std::unique_ptr<MmapWrapper> mmapped = MmapWrapper::create(file_path, writable_mmap);
            if (mmapped) {
                uint8_t* mmap_data             = static_cast<uint8_t*>(mmapped->writable_data());
                ggml_backend_buffer_t buf_mmap = ggml_backend_cpu_buffer_from_ptr(mmap_data, mmapped->size());
                if (buf_mmap) {
                    LOG_INFO("using mmap for '%s'", file_path.c_str());
                    fdata.mmbuffer = std::shared_ptr<struct ggml_backend_buffer>(buf_mmap, ggml_backend_buffer_free);
                } else {
                    LOG_WARN("mmap: failed to create backend buffer for file %s", fdata.path.c_str());
                }
                fdata.mmapped = std::shared_ptr<MmapWrapper>(std::move(mmapped));
            } else {
                LOG_WARN("failed to memory-map '%s' (falling back to read())", file_path.c_str());
            }
        } else if (!is_zip) {
            LOG_INFO("NOT using mmap for '%s' (mmap disabled by caller)",
                     file_path.c_str());
        }

        file_data.push_back(std::move(fdata));
    }

    model_files_processed = true;

    int64_t end_time        = ggml_time_ms();
    int64_t process_time_ms = end_time - start_time;

    LOG_INFO("model files processing completed in %.2fs", process_time_ms / 1000.f);
}

std::vector<MmapTensorStore> ModelLoader::mmap_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                                       std::set<std::string> ignore_tensors,
                                                       bool writable_mmap) {
    process_model_files(true, writable_mmap);

    std::vector<MmapTensorStore> result;
    uint64_t mapped_bytes = 0;
    size_t mapped_tensors = 0;

    LOG_DEBUG("memory-mapping tensors...");

    int64_t t_start = ggml_time_ms();

    for (auto& fdata : file_data) {
        if (!fdata.mmbuffer)
            continue;

        const std::vector<TensorStorage>& file_tensors = fdata.tensors;

        size_t file_mapped_bytes   = 0;
        size_t file_mapped_tensors = 0;

        for (const auto& tensor_storage : file_tensors) {
            const std::string& name = tensor_storage.name;

            bool is_ignored = false;
            for (const auto& ignore_prefix : ignore_tensors) {
                if (starts_with(name, ignore_prefix)) {
                    is_ignored = true;
                    break;
                }
            }
            if (is_ignored)
                continue;

            auto it = tensors.find(name);
            if (it == tensors.end())
                continue;

            ggml_tensor* dst_tensor = it->second;
            if (dst_tensor == nullptr)
                continue;

            if (tensor_storage.type != dst_tensor->type)
                continue;

            size_t tensor_size   = tensor_storage.nbytes();
            size_t tensor_offset = tensor_storage.offset;

            if (tensor_storage.ne[0] != dst_tensor->ne[0] ||
                tensor_storage.ne[1] != dst_tensor->ne[1] ||
                tensor_storage.ne[2] != dst_tensor->ne[2] ||
                tensor_storage.ne[3] != dst_tensor->ne[3] ||
                tensor_size != ggml_nbytes(dst_tensor)) {
                // let load_tensors worry about this
                continue;
            }

            ggml_backend_buffer_t buf_mmap = fdata.mmbuffer.get();
            uint8_t* mmap_data             = static_cast<uint8_t*>(ggml_backend_buffer_get_base(buf_mmap));
            dst_tensor->buffer             = buf_mmap;
            dst_tensor->data               = mmap_data + tensor_offset;

            file_mapped_bytes += tensor_size;
            file_mapped_tensors++;
        }

        if (file_mapped_bytes > 0) {
            mapped_tensors += file_mapped_tensors;
            mapped_bytes += file_mapped_bytes;
            result.push_back({fdata.mmapped, fdata.mmbuffer});
        }
    }

    int64_t t_end       = ggml_time_ms();
    int64_t duration_ms = t_end - t_start;

    LOG_INFO("memory-mapped %zu tensors in %zu files (%.2f MB), taking %.2fs",
             mapped_tensors,
             result.size(),
             mapped_bytes / (1024.0 * 1024.0),
             duration_ms / 1000.0);

    return result;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads_p, bool enable_mmap) {
    process_model_files(enable_mmap, false);

    std::atomic<int64_t> read_time_ms(0);
    std::atomic<int64_t> memcpy_time_ms(0);
    std::atomic<int64_t> copy_to_backend_time_ms(0);
    std::atomic<int64_t> convert_time_ms(0);
    std::atomic<uint64_t> bytes_processed(0);

    int num_threads_to_use = n_threads_p > 0 ? n_threads_p : sd_get_num_physical_cores();
    LOG_DEBUG("using %d threads for model loading", num_threads_to_use);

    int64_t start_time = ggml_time_ms();

    size_t total_tensors_to_process = 0;
    for (const auto& fdata : file_data) {
        total_tensors_to_process += fdata.tensors.size();
    }

    bool success                   = true;
    size_t total_tensors_processed = 0;
    const int64_t t_start          = start_time;
    int last_n_threads             = 1;

    for (auto& fdata : file_data) {
        const std::string& file_path = fdata.path;
        LOG_DEBUG("loading tensors from %s", file_path.c_str());

        const std::vector<TensorStorage>& file_tensors = fdata.tensors;

        bool is_zip = fdata.is_zip;

        std::shared_ptr<MmapWrapper> mmapped = fdata.mmapped;

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
                zip_t* zip = nullptr;
                if (is_zip) {
                    zip = zip_open(file_path.c_str(), 0, 'r');
                    if (zip == nullptr) {
                        LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                } else if (!mmapped) {
                    file.open(file_path, std::ios::binary);
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

                    const TensorStorage& tensor_storage = file_tensors[idx];
                    ggml_tensor* dst_tensor             = nullptr;

                    t0 = ggml_time_ms();

                    if (!on_new_tensor_cb(tensor_storage, &dst_tensor)) {
                        LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                        failed = true;
                        break;
                    }

                    if (dst_tensor == nullptr) {
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);
                        continue;
                    }

                    // skip mmapped tensors
                    if (dst_tensor->buffer != nullptr && dst_tensor->buffer == fdata.mmbuffer.get()) {
                        continue;
                    }

                    size_t nbytes_to_read = tensor_storage.nbytes_to_read();

                    auto read_data = [&](char* buf, size_t n) {
                        if (zip != nullptr) {
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
                        } else if (mmapped) {
                            if (!mmapped->copy_data(buf, n, tensor_storage.offset)) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                failed = true;
                            }
                        } else {
                            file.seekg(tensor_storage.offset);
                            file.read(buf, n);
                            if (!file) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                failed = true;
                            }
                        }
                    };

                    char* read_buf    = nullptr;
                    char* target_buf  = nullptr;
                    char* convert_buf = nullptr;
                    if (dst_tensor->buffer == nullptr || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        if (tensor_storage.type == dst_tensor->type) {
                            GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                            if (tensor_storage.is_f64 || tensor_storage.is_i64) {
                                read_buffer.resize(tensor_storage.nbytes_to_read());
                                read_buf = (char*)read_buffer.data();
                            } else {
                                read_buf = (char*)dst_tensor->data;
                            }
                            target_buf = (char*)dst_tensor->data;
                        } else {
                            read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                            read_buf    = (char*)read_buffer.data();
                            target_buf  = read_buf;
                            convert_buf = (char*)dst_tensor->data;
                        }
                    } else {
                        read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                        read_buf   = (char*)read_buffer.data();
                        target_buf = read_buf;

                        if (tensor_storage.type != dst_tensor->type) {
                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_buf = (char*)convert_buffer.data();
                        }
                    }

                    t0 = ggml_time_ms();
                    read_data(read_buf, nbytes_to_read);
                    t1 = ggml_time_ms();
                    read_time_ms.fetch_add(t1 - t0);

                    t0 = ggml_time_ms();
                    if (tensor_storage.is_f8_e4m3) {
                        f8_e4m3_to_f16_vec((uint8_t*)read_buf, (uint16_t*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        f8_e5m2_to_f16_vec((uint8_t*)read_buf, (uint16_t*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_f64) {
                        f64_to_f32_vec((double*)read_buf, (float*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_i64) {
                        i64_to_i32_vec((int64_t*)read_buf, (int32_t*)target_buf, tensor_storage.nelements());
                    }
                    if (tensor_storage.type != dst_tensor->type) {
                        if (convert_buf == nullptr) {
                            LOG_ERROR("read tensor data failed: too less memory for conversion");
                            failed = true;
                            return;
                        }
                        convert_tensor((void*)target_buf,
                                       tensor_storage.type,
                                       convert_buf,
                                       dst_tensor->type,
                                       (int)tensor_storage.nelements() / (int)tensor_storage.ne[0],
                                       (int)tensor_storage.ne[0]);
                    } else {
                        convert_buf = read_buf;
                    }
                    t1 = ggml_time_ms();
                    convert_time_ms.fetch_add(t1 - t0);

                    if (dst_tensor->buffer != nullptr && !ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        t0 = ggml_time_ms();
                        ggml_backend_tensor_set(dst_tensor, convert_buf, 0, ggml_nbytes(dst_tensor));
                        t1 = ggml_time_ms();
                        copy_to_backend_time_ms.fetch_add(t1 - t0);
                    }

                    bytes_processed.fetch_add((uint64_t)nbytes_to_read);
                }
                if (zip != nullptr) {
                    zip_close(zip);
                }
            });
        }

        while (true) {
            size_t current_idx = tensor_idx.load();
            if (current_idx >= file_tensors.size() || failed) {
                break;
            }
            size_t curr_num       = total_tensors_processed + current_idx;
            float elapsed_seconds = (ggml_time_ms() - t_start) / 1000.0f;
            pretty_bytes_progress(static_cast<int>(curr_num),
                                  static_cast<int>(total_tensors_to_process),
                                  bytes_processed.load(),
                                  elapsed_seconds);
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
        pretty_bytes_progress(static_cast<int>(total_tensors_processed),
                              static_cast<int>(total_tensors_to_process),
                              bytes_processed.load(),
                              (ggml_time_ms() - t_start) / 1000.0f);
        if (total_tensors_processed < total_tensors_to_process) {
            printf("\n");
        }
    }

    int64_t end_time = ggml_time_ms();
    LOG_INFO("loading tensors completed, taking %.2fs (read: %.2fs, memcpy: %.2fs, convert: %.2fs, copy_to_backend: %.2fs)",
             (end_time - start_time) / 1000.f,
             (read_time_ms.load() / (float)last_n_threads) / 1000.f,
             (memcpy_time_ms.load() / (float)last_n_threads) / 1000.f,
             (convert_time_ms.load() / (float)last_n_threads) / 1000.f,
             (copy_to_backend_time_ms.load() / (float)last_n_threads) / 1000.f);
    return success;
}

bool ModelLoader::load_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               std::set<std::string> ignore_tensors,
                               int n_threads,
                               bool enable_mmap) {
    std::set<std::string> tensor_names_in_file;
    std::mutex tensor_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
        {
            std::lock_guard<std::mutex> lock(tensor_names_mutex);
            tensor_names_in_file.insert(name);
        }

        ggml_tensor* real;
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

    bool success = load_tensors(on_new_tensor_cb, n_threads, enable_mmap);
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

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) {
    size_t alignment = 128;
    if (backend != nullptr) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    std::vector<TensorStorage> processed_tensor_storages;
    for (auto [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}

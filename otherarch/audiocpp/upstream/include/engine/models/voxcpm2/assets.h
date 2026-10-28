#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxcpm2 {

struct VoxCPM2RopeScalingConfig {
    std::string type;
    std::vector<float> long_factor;
    std::vector<float> short_factor;
    int64_t original_max_position_embeddings = 0;
};

struct VoxCPM2MiniCPMConfig {
    int64_t bos_token_id = 1;
    int64_t eos_token_id = 2;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t kv_channels = 0;
    int64_t vocab_size = 0;
    int64_t scale_emb = 1;
    int64_t dim_model_base = 0;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 10000.0F;
    float scale_depth = 1.0F;
    bool use_mup = false;
    bool no_rope = false;
    VoxCPM2RopeScalingConfig rope_scaling;
};

struct VoxCPM2LocalTransformerConfig {
    int64_t hidden_dim = 0;
    int64_t ffn_dim = 0;
    int64_t num_heads = 0;
    int64_t num_layers = 0;
    int64_t kv_channels = 0;
};

struct VoxCPM2CFMConfig {
    float sigma_min = 1.0e-6F;
    std::string solver = "euler";
    std::string t_scheduler = "log-norm";
    float inference_cfg_rate = 2.0F;
};

struct VoxCPM2DiTConfig : VoxCPM2LocalTransformerConfig {
    bool mean_mode = false;
    VoxCPM2CFMConfig cfm;
};

struct VoxCPM2AudioVAEConfig {
    int64_t encoder_dim = 0;
    std::vector<int64_t> encoder_rates;
    int64_t latent_dim = 0;
    int64_t decoder_dim = 0;
    std::vector<int64_t> decoder_rates;
    std::vector<int64_t> sample_rate_bin_boundaries;
    int sample_rate = 0;
    int output_sample_rate = 0;
};

struct VoxCPM2Config {
    std::string architecture;
    VoxCPM2MiniCPMConfig lm;
    int64_t patch_size = 4;
    int64_t feat_dim = 64;
    int64_t residual_lm_num_layers = 8;
    bool residual_lm_no_rope = false;
    int64_t scalar_quantization_latent_dim = 512;
    int64_t scalar_quantization_scale = 9;
    VoxCPM2LocalTransformerConfig encoder;
    VoxCPM2DiTConfig dit;
    VoxCPM2AudioVAEConfig audio_vae;
    int64_t max_length = 8192;
    std::string device = "cuda";
    std::string dtype = "bfloat16";
};

struct VoxCPM2Assets {
    assets::ResourceBundle resources;
    VoxCPM2Config config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audiovae_weights;
};

std::shared_ptr<const VoxCPM2Assets> load_voxcpm2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::voxcpm2


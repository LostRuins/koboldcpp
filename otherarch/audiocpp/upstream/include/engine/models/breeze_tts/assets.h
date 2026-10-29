#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::breeze_tts {

struct BreezeTTSConfig {
    int sample_rate = 24000;
    int64_t hidden_size = 2048;
    int64_t intermediate_size = 6144;
    int64_t layers = 28;
    int64_t heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t vocab_size = 2051;
    int64_t lm_head_size = 2052;
    int64_t text_vocab_size = 262158;
    int64_t num_codebooks = 16;
    int64_t max_position_embeddings = 2048;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 500000.0F;
    float rope_scaling_factor = 32.0F;
    float rope_low_freq_factor = 0.125F;
    float rope_high_freq_factor = 0.5F;
    int64_t rope_original_max_position_embeddings = 1024;
    bool rope_scaling_enabled = false;
    int64_t audio_token_id = 262144;
    int64_t audio_eos_token_id = 262145;
    int64_t codebook_pad_token_id = 2050;
    int64_t codebook_eos_token_id = 0;

    int64_t text_hidden_size = 1152;
    int64_t text_intermediate_size = 6912;
    int64_t text_layers = 26;
    int64_t text_heads = 4;
    int64_t text_kv_heads = 1;
    int64_t text_head_dim = 256;
    int64_t text_max_position_embeddings = 32768;
    float text_rms_norm_eps = 1.0e-6F;
    float text_rope_theta = 1000000.0F;
    float text_rope_linear_factor = 8.0F;
    float text_query_pre_attn_scalar = 256.0F;
    std::vector<float> text_layer_rope_theta;
    std::vector<float> text_layer_rope_freq_scale;

    int64_t depth_hidden_size = 1024;
    int64_t depth_intermediate_size = 4096;
    int64_t depth_layers = 4;
    int64_t depth_heads = 8;
    int64_t depth_kv_heads = 2;
    int64_t depth_head_dim = 128;
    float depth_rms_norm_eps = 1.0e-5F;
    float depth_rope_theta = 500000.0F;
    float depth_rope_scaling_factor = 32.0F;
    float depth_rope_low_freq_factor = 0.001953125F;
    float depth_rope_high_freq_factor = 0.0078125F;
    int64_t depth_rope_original_max_position_embeddings = 16;
    bool depth_rope_scaling_enabled = false;
};

struct BreezeTTSAssets {
    std::filesystem::path model_root;
    engine::assets::ResourceBundle resources;
    BreezeTTSConfig config;
    std::shared_ptr<const engine::assets::TensorSource> weights;
};

std::shared_ptr<const BreezeTTSAssets> load_breeze_tts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::breeze_tts

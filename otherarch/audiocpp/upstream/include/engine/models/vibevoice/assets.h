#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoiceTokenizerConfig {
    int64_t channels = 1;
    bool causal = true;
    int64_t vae_dim = 0;
    float fix_std = 0.0F;
    std::string std_dist_type;
    std::string mixer_layer;
    std::string conv_norm;
    std::string pad_mode;
    bool disable_last_norm = true;
    std::string layernorm;
    float layernorm_eps = 1.0e-5F;
    bool layernorm_elementwise_affine = true;
    bool conv_bias = true;
    float layer_scale_init_value = 1.0e-6F;
    float weight_init_value = 1.0e-2F;
    int64_t encoder_n_filters = 0;
    std::vector<int64_t> encoder_ratios;
    std::string encoder_depths;
    int64_t decoder_n_filters = 0;
    std::vector<int64_t> decoder_ratios;
    std::string decoder_depths;
};

struct VibeVoiceQwen2Config {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t max_window_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    int64_t head_dim = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
    bool use_cache = true;
    bool use_sliding_window = false;
};

struct VibeVoiceDiffusionHeadConfig {
    int64_t ddpm_batch_mul = 4;
    std::string ddpm_beta_schedule;
    int64_t ddpm_num_inference_steps = 20;
    int64_t ddpm_num_steps = 1000;
    std::string diffusion_type;
    float head_ffn_ratio = 3.0F;
    int64_t head_layers = 4;
    int64_t hidden_size = 0;
    int64_t latent_size = 0;
    std::string prediction_type;
    float rms_norm_eps = 1.0e-5F;
    int64_t speech_vae_dim = 0;
};

struct VibeVoiceConfig {
    int64_t acoustic_vae_dim = 0;
    int64_t semantic_vae_dim = 0;
    std::string model_type;
    std::string torch_dtype;
    VibeVoiceTokenizerConfig acoustic_tokenizer;
    VibeVoiceTokenizerConfig semantic_tokenizer;
    VibeVoiceQwen2Config decoder;
    VibeVoiceDiffusionHeadConfig diffusion_head;
};

struct VibeVoiceAudioProcessorConfig {
    int sample_rate = 24000;
    bool normalize_audio = true;
    float target_db_fs = -25.0F;
    float eps = 1.0e-6F;
};

struct VibeVoiceProcessorConfig {
    int64_t speech_tok_compress_ratio = 3200;
    bool db_normalize = true;
    std::string language_model_pretrained_name = "Qwen/Qwen2.5-1.5B";
    VibeVoiceAudioProcessorConfig audio_processor;
};

struct VibeVoiceAssets {
    assets::ResourceBundle resources;
    VibeVoiceConfig config;
    VibeVoiceProcessorConfig processor;
    std::shared_ptr<const assets::TensorSource> model_weights;
    float speech_scaling_factor = 1.0F;
    float speech_bias_factor = 0.0F;
    // Set once a fine-tune adapter has been overlaid onto model_weights, so the same adapter is
    // not applied twice (e.g. via both --load-option and --session-option).
    bool fine_tune_applied = false;
};

std::shared_ptr<const VibeVoiceAssets> load_vibevoice_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::vibevoice

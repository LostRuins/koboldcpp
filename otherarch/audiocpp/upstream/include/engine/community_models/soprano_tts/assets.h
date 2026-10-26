#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::community_models::soprano_tts {

struct SopranoTTSConfig {
    // Qwen3 causal LM (config.json).
    int64_t hidden_size = 512;
    int64_t intermediate_size = 2304;
    int64_t layers = 17;
    int64_t attention_heads = 4;
    int64_t kv_heads = 1;
    int64_t head_dim = 128;
    int64_t vocab_size = 8192;
    int64_t max_position_embeddings = 1024;
    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000.0f;
    int32_t bos_token_id = 3;
    int32_t eos_token_id = 3;

    // Non-iterative Vocos decoder (decoder.pth / config).
    int64_t decoder_input_channels = 512;   // == hidden_size
    int64_t decoder_dim = 768;
    int64_t decoder_intermediate_dim = 2304;
    int64_t decoder_num_layers = 8;
    int64_t dw_kernel = 3;
    int64_t n_fft = 2048;
    int64_t hop_length = 512;
    int64_t upscale = 4;
    int64_t sample_rate = 32000;
    int64_t token_size = 2048;              // samples per generated frame
    int64_t max_new_tokens = 128;
    float temperature = 0.3f;
    float top_p = 0.95f;
    float repetition_penalty = 1.2f;
};

struct SopranoTTSAssets {
    assets::ResourceBundle resources;
    SopranoTTSConfig config;
    std::shared_ptr<const assets::TensorSource> backbone_weights;
    std::shared_ptr<const assets::TensorSource> decoder_weights;
};

struct SopranoGenerationOptions {
    // Per-chunk limit; the official reference allows up to 512 frames
    // (32 s of audio) per sentence.
    int64_t max_new_tokens = 512;
    float temperature = 0.3f;
    float top_p = 0.95f;
    float repetition_penalty = 1.2f;
    uint64_t seed = 0;
    bool has_seed = false;
    // Additive bias on the EOS logit; 0 disables (opt-in runaway mitigation).
    float eos_bias = 0.0F;
};

std::shared_ptr<const SopranoTTSAssets> load_soprano_tts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::community_models::soprano_tts
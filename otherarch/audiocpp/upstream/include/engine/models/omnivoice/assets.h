#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

struct OmniVoiceQwen3DiffusionConfig {
    std::string model_type;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct OmniVoiceHiggsAudioV2TokenizerConfig {
    struct SemanticModelConfig {
        int64_t hidden_size = 0;
        int64_t intermediate_size = 0;
        int64_t num_attention_heads = 0;
        int64_t num_hidden_layers = 0;
        int64_t num_conv_pos_embeddings = 0;
        int64_t num_conv_pos_embedding_groups = 0;
        float layer_norm_eps = 1.0e-5F;
        bool feat_proj_layer_norm = true;
        bool do_stable_layer_norm = false;
        std::vector<int64_t> conv_dim;
        std::vector<int64_t> conv_kernel;
        std::vector<int64_t> conv_stride;
    };

    struct AcousticModelConfig {
        int64_t codebook_dim = 0;
        int64_t encoder_hidden_size = 0;
        int64_t decoder_hidden_size = 0;
        int64_t hidden_size = 0;
        std::vector<int64_t> downsampling_ratios;
        std::vector<int64_t> upsampling_ratios;
    };

    std::string model_type;
    int sample_rate = 24000;
    int semantic_sample_rate = 16000;
    int64_t downsample_factor = 0;
    int64_t codebook_size = 0;
    int64_t num_codebooks = 0;
    int64_t acoustic_codebooks = 0;
    int64_t hop_length = 0;
    int64_t hidden_size = 0;
    int64_t codebook_dim = 0;
    int64_t kernel_size = 0;
    int64_t unit_kernel_size = 0;
    std::vector<float> target_bandwidths;
    std::vector<int64_t> channel_ratios;
    std::vector<int64_t> strides;
    std::vector<int64_t> block_dilations;
    SemanticModelConfig semantic_model;
    AcousticModelConfig acoustic_model;
};

struct OmniVoiceConfig {
    std::string model_type;
    int64_t audio_vocab_size = 0;
    int64_t audio_mask_id = 0;
    int64_t num_audio_codebook = 0;
    std::vector<float> audio_codebook_weights;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = 0;
    OmniVoiceQwen3DiffusionConfig llm;
    OmniVoiceHiggsAudioV2TokenizerConfig audio_tokenizer;
    std::vector<std::string> supported_languages = {"Auto"};
};

struct OmniVoiceAssets {
    assets::ResourceBundle resources;
    OmniVoiceConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const OmniVoiceAssets> load_omnivoice_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::omnivoice

#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::neutts {

struct NeuTTSBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    int64_t max_context = 2048;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    int32_t eos_token_id = 151674;
    int32_t pad_token_id = 151645;
    std::vector<std::string> supported_languages;
    std::vector<std::string> supported_emotions;
};

struct NeuTTSCodecConfig {
    int64_t sample_rate = 16000;
    int64_t output_sample_rate = 24000;
    int64_t hop_length = 480;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t quantization_dim = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::vector<int64_t> quantization_levels;
};

struct NeuTTSSpeakerPrompt {
    std::string id;
    std::string reference_text;
    std::vector<int32_t> speech_codes;
};

struct NeuTTSAssets {
    assets::ResourceBundle resources;
    NeuTTSBackboneConfig backbone;
    NeuTTSCodecConfig codec;
    std::shared_ptr<const assets::TensorSource> backbone_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
    std::unordered_map<std::string, NeuTTSSpeakerPrompt> speakers;
};

std::shared_ptr<const NeuTTSAssets> load_neutts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::neutts

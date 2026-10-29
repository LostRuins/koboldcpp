#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace engine::models::qwen3_tts {

struct Qwen3TTSTalkerConfig {
    int64_t max_position_embeddings = 32768;
    int64_t hidden_size = 0;
    int64_t text_hidden_size = 0;
    int64_t text_vocab_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t num_code_groups = 0;
    int64_t vocab_size = 0;
    int64_t codec_bos_id = 0;
    int64_t codec_eos_token_id = 0;
    int64_t codec_think_id = 0;
    int64_t codec_nothink_id = 0;
    int64_t codec_pad_id = 0;
    int64_t codec_think_bos_id = 0;
    int64_t codec_think_eos_id = 0;
    std::unordered_map<std::string, int64_t> codec_language_id;
    std::unordered_map<std::string, int64_t> speaker_id;
    std::unordered_map<std::string, std::optional<std::string>> speaker_dialect;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct Qwen3TTSCodePredictorConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct Qwen3TTSSpeechTokenizerConfig {
    std::string model_type;
    int input_sample_rate = 0;
    int output_sample_rate = 0;
    int64_t num_quantizers = 0;
    int64_t codebook_size = 0;
    int64_t semantic_codebook_size = 0;
};

struct Qwen3TTSEcapaTdnnEncoderConfig {
    int64_t embedding_dim = 0;
    int sample_rate = 0;
};

struct Qwen3TTSConfig {
    Qwen3TTSVariant variant = Qwen3TTSVariant::Base;
    std::string tts_model_type;
    std::string tts_model_size;
    std::string tokenizer_type;
    int64_t max_new_tokens = 2048;
    Qwen3TTSTalkerConfig talker;
    Qwen3TTSCodePredictorConfig code_predictor;
    Qwen3TTSSpeechTokenizerConfig speech_tokenizer;
    Qwen3TTSEcapaTdnnEncoderConfig speaker_encoder;
    int64_t tts_bos_token_id = 0;
    int64_t tts_eos_token_id = 0;
    int64_t tts_pad_token_id = 0;
    bool has_speaker_encoder = false;
};

struct Qwen3TTSAssets {
    assets::ResourceBundle resources;
    Qwen3TTSConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> speech_tokenizer_weights;
};

std::shared_ptr<const Qwen3TTSAssets> load_qwen3_tts_assets(const std::filesystem::path & model_path);

std::shared_ptr<const Qwen3TTSAssets> apply_qwen3_tts_lora(
    std::shared_ptr<const Qwen3TTSAssets> assets,
    const std::unordered_map<std::string, std::string> & options);

}  // namespace engine::models::qwen3_tts

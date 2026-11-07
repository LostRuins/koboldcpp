#pragma once

#include "engine/models/qwen3_asr/assets.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::audio8_asr {

namespace qwen3_asr = engine::models::qwen3_asr;

struct Audio8ASRDecoderConfig {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 32768;
    int64_t audio_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t max_new_tokens = 256;
    std::vector<int64_t> eos_token_ids;
    bool tie_word_embeddings = true;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

// The MLP tower + projector that adapts the Qwen3-ASR encoder output to the
// Audio8 decoder hidden size: four pre-norm residual MLP blocks, a final norm,
// an adaptive average pool down to the prompt's audio token count, and a
// LayerNorm + linear projector into the decoder hidden size.
struct Audio8TowerConfig {
    int64_t input_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 4;
    int64_t output_size = 0;
    float norm_eps = 1.0e-5F;
};

struct Audio8ASRConfig {
    int64_t merge_factor = 4;
    int64_t max_audio_samples = 480000;
    int64_t user_token_id = 0;
    int64_t begin_audio_token_id = 0;
    int64_t end_audio_token_id = 0;
    int64_t assistant_token_id = 0;
    qwen3_asr::Qwen3ASRFrontendConfig frontend;
    qwen3_asr::Qwen3ASRAudioEncoderConfig audio_encoder;
    Audio8TowerConfig tower;
    Audio8ASRDecoderConfig text_decoder;
    std::vector<std::string> supported_languages;
};

struct Audio8ASRPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_token_positions;
};

struct Audio8ASRAudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct Audio8ASRGenerationOptions {
    int64_t max_new_tokens = 256;
};

struct Audio8ASRGeneratedTokens {
    std::vector<int32_t> token_ids;
};

// The reference processor hands the model a bfloat16 mel tensor; fp32
// pipelines replicate that rounding before encoding (round to nearest even,
// like torch's .to(torch.bfloat16)).
inline float audio8_asr_round_f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    bits &= 0xFFFF0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Audio token count used by the Arkasr processor:
//   mel_frames = samples // hop_length
//   downsampled = (mel_frames + 1) // 2
//   tokens = max(downsampled // merge_factor, 1)
inline int64_t audio8_asr_prompt_audio_token_count(
    int64_t mel_frames,
    int64_t merge_factor) {
    if (mel_frames <= 0) {
        throw std::runtime_error("Audio8 ASR requires positive mel frame count");
    }
    if (merge_factor <= 0) {
        throw std::runtime_error("Audio8 ASR merge_factor must be positive");
    }
    const int64_t downsampled = (mel_frames + 1) / 2;
    const int64_t merged = downsampled / merge_factor;
    return merged > 0 ? merged : 1;
}

}  // namespace engine::community_models::audio8_asr

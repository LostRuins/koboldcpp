#pragma once

#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::fish_audio {

struct FishAudioGenerationOptions {
    int64_t max_new_tokens = 1024;
    int64_t text_chunk_size = 200;
    float top_p = 0.8F;
    int top_k = 30;
    float temperature = 0.8F;
    uint32_t seed = 1234;
};

struct FishAudioReference {
    std::optional<runtime::AudioBuffer> audio = std::nullopt;
    std::string text;
    std::string cache_id;
};

struct FishAudioRequest {
    std::string text;
    std::vector<FishAudioReference> references;
    FishAudioGenerationOptions generation;
};

struct FishAudioConversationTurn {
    std::string text;
    engine::codecs::FishDacCodes codes;
};

struct FishAudioPrompt {
    std::vector<int32_t> matrix;
    int64_t codebook_rows = 0;
    int64_t steps = 0;
    std::string text;
};

struct FishAudioTextConfig {
    int64_t vocab_size = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = true;
    bool attention_qk_norm = true;
};

struct FishAudioFastConfig {
    int64_t vocab_size = 0;
    int64_t num_codebooks = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = false;
    bool attention_qk_norm = false;
};

struct FishAudioConfig {
    std::string model_type;
    std::string torch_dtype;
    int64_t semantic_start_token_id = 0;
    int64_t semantic_end_token_id = 0;
    int64_t im_end_token_id = 0;
    bool norm_fastlayer_input = false;
    FishAudioTextConfig text;
    FishAudioFastConfig fast;
    engine::codecs::FishDacCodecConfig codec;
};

}  // namespace engine::models::fish_audio

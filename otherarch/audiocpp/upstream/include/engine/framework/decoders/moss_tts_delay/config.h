#pragma once

// Geometry and token ids for the MOSS `moss_tts_delay` family.
//
// One architecture covers several checkpoints, differing mainly in size and in
// how many codebooks they emit: MOSS-VoiceGenerator (Qwen3-1.7B, n_vq 16),
// MOSS-TTSD (8B, n_vq 16) and MOSS-TTS-v1.5 (8B, n_vq 32). Every field here is
// read from the checkpoint's config.json, so a new member of the family is a
// configuration rather than a code change.
//
// Promoted out of the moss_voicegen model in the same spirit as the MOSS audio
// tokenizer codec runtime, so the family shares one set of runtimes instead of
// copying them per checkpoint.

#include "engine/framework/io/json.h"

#include <cstdint>
#include <string_view>

namespace engine::decoders {

// Qwen3 backbone geometry, read from the checkpoint's "language_config" block.
struct MossTtsDelayBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
};

struct MossTtsDelayConfig {
    MossTtsDelayBackboneConfig backbone;
    // The family emits 1 + n_vq ids per step: one text id and one code per
    // codebook. 16 for VoiceGenerator and TTSD, 32 for MOSS-TTS-v1.5.
    int64_t num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    int64_t audio_pad_code = 0;
    // Some checkpoints' config.json omits these; MossTTSDelayConfig's defaults
    // apply and match what the tokenizer resolves for <|im_start|>/<|im_end|>.
    int64_t pad_token_id = 151643;
    int64_t im_start_token_id = 151644;
    int64_t im_end_token_id = 151645;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t audio_user_slot_token_id = 0;
    int64_t audio_assistant_gen_slot_token_id = 0;
    int64_t audio_assistant_delay_slot_token_id = 0;
    int64_t sampling_rate = 0;
};

// Parses a moss_tts_delay checkpoint's config.json. `model_label` only names the
// model in error messages; every field read here is the family's.
MossTtsDelayConfig parse_moss_tts_delay_config(const io::json::Value & root, std::string_view model_label);

}  // namespace engine::decoders

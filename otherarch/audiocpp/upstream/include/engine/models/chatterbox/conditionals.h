#pragma once

#include "engine/models/chatterbox/components.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/codecs/s3gen_runtime.h"

#include <cstdint>
#include <vector>

namespace engine::models::chatterbox {

struct ChatterboxPromptPrepConfig {
    int64_t encoder_condition_samples = 6 * 16000;
    int64_t decoder_condition_samples = 10 * 24000;
    int64_t t3_speech_cond_prompt_len = 150;
    int64_t tokenizer_sample_rate = 16000;
    int64_t generator_sample_rate = 24000;
};

struct ChatterboxT3Conditionals {
    std::vector<float> speaker_embedding;
    int64_t speaker_embedding_size = 0;
    std::vector<int32_t> cond_prompt_speech_tokens;
    int64_t cond_prompt_token_count = 0;
    std::vector<float> emotion_adv;
    int64_t emotion_adv_count = 0;
};

struct ChatterboxConditionalsOutputs {
    ChatterboxT3Conditionals t3;
    EmbedReferenceOutputs gen;
    double gen_ms = 0.0;
    double voice_encoder_ms = 0.0;
    double tokenizer_ms = 0.0;
};

class ChatterboxConditionalsComponent {
public:
    ChatterboxConditionalsComponent(
        engine::models::chatterbox::VoiceEncoderComponent voice_encoder,
        engine::models::chatterbox::S3TokenizerComponent tokenizer,
        engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
        ChatterboxPromptPrepConfig config = {});

    const ChatterboxPromptPrepConfig & config() const noexcept;

    ChatterboxConditionalsOutputs prepare(
        const runtime::AudioBuffer & audio,
        float exaggeration = 0.5f) const;

private:
    engine::models::chatterbox::VoiceEncoderComponent voice_encoder_;
    engine::models::chatterbox::S3TokenizerComponent tokenizer_;
    engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder_;
    ChatterboxPromptPrepConfig config_;
};

}  // namespace engine::models::chatterbox

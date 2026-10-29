#pragma once

#include "engine/models/qwen3_tts/speaker_encoder.h"
#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/tokenizer_speech_encoder.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"
#include "engine/models/qwen3_tts/types.h"

#include <optional>

namespace engine::models::qwen3_tts {

struct Qwen3VoiceClonePrompt {
    Qwen3SpeakerEmbedding speaker_embedding;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::vector<int32_t> reference_text_ids;
    bool icl_mode = true;
};

class Qwen3TTSVoiceClonePromptBuilder {
public:
    Qwen3TTSVoiceClonePromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        const Qwen3SpeechTokenizerEncoderRuntime & speech_encoder,
        const Qwen3TTSEcapaTdnnEncoderRuntime & speaker_encoder,
        int64_t text_token_limit);

    Qwen3VoiceClonePrompt build_voice_prompt(const Qwen3VoiceCloneInput & input) const;
    Qwen3TalkerPrefill build_prefill(const Qwen3TTSRequest & request, const Qwen3VoiceClonePrompt & prompt) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    const Qwen3SpeechTokenizerEncoderRuntime & speech_encoder_;
    const Qwen3TTSEcapaTdnnEncoderRuntime & speaker_encoder_;
    int64_t text_token_limit_ = 0;
};

}  // namespace engine::models::qwen3_tts

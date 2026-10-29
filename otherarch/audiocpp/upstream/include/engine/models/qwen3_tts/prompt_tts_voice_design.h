#pragma once

#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"
#include "engine/models/qwen3_tts/types.h"

namespace engine::models::qwen3_tts {

class Qwen3TTSVoiceDesignPromptBuilder {
public:
    Qwen3TTSVoiceDesignPromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        int64_t text_token_limit,
        int64_t instruction_token_limit);

    Qwen3TalkerPrefill build_prefill(const Qwen3TTSRequest & request) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    int64_t text_token_limit_ = 0;
    int64_t instruction_token_limit_ = 0;
};

}  // namespace engine::models::qwen3_tts

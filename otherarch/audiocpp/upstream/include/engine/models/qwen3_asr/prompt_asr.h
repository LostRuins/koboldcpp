#pragma once

#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_asr/types.h"

namespace engine::models::qwen3_asr {

class Qwen3ASRPromptBuilder {
public:
    explicit Qwen3ASRPromptBuilder(const Qwen3ASRTextTokenizer & tokenizer);

    Qwen3ASRPrompt build(const Qwen3ASRRequest & request, int64_t audio_feature_tokens) const;

private:
    const Qwen3ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::models::qwen3_asr

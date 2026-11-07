#include "engine/models/qwen3_asr/prompt_asr.h"

namespace engine::models::qwen3_asr {

Qwen3ASRPromptBuilder::Qwen3ASRPromptBuilder(const Qwen3ASRTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

Qwen3ASRPrompt Qwen3ASRPromptBuilder::build(const Qwen3ASRRequest & request, int64_t audio_feature_tokens) const {
    return tokenizer_.build_prompt(request.context, request.language, audio_feature_tokens);
}

}  // namespace engine::models::qwen3_asr

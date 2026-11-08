#pragma once

#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_asr/types.h"

namespace engine::models::qwen3_asr {

class Qwen3ASRPostprocessor {
public:
    explicit Qwen3ASRPostprocessor(const Qwen3ASRTextTokenizer & tokenizer);

    Qwen3ASRResult decode(const Qwen3ASRGeneratedTokens & tokens, const Qwen3ASRRequest & request) const;

private:
    const Qwen3ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::models::qwen3_asr

#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_asr/types.h"

#include <string>
#include <vector>

namespace engine::models::qwen3_forced_aligner {

std::vector<std::string> tokenize_alignable_words(
    const std::string & text,
    const std::string & language);

bool has_alignable_words(
    const std::string & text,
    const std::string & language);

struct ForcedAlignPrompt {
    engine::models::qwen3_asr::Qwen3ASRPrompt prompt;
    std::vector<std::string> words;
    std::vector<int32_t> timestamp_positions;
};

class Qwen3ForcedAlignProcessor {
public:
    Qwen3ForcedAlignProcessor(
        const engine::models::qwen3_asr::Qwen3ASRAssets & assets,
        const engine::models::qwen3_asr::Qwen3ASRTextTokenizer & tokenizer);

    ForcedAlignPrompt build_prompt(
        const std::string & text,
        const std::string & language,
        int64_t audio_feature_tokens) const;

    std::vector<engine::runtime::WordTimestamp> parse_timestamps(
        const std::vector<std::string> & words,
        const std::vector<int32_t> & timestamp_ids,
        int sample_rate,
        int64_t audio_frames,
        bool clamp_timestamps_to_audio) const;

private:
    const engine::models::qwen3_asr::Qwen3ASRAssets & assets_;
    const engine::models::qwen3_asr::Qwen3ASRTextTokenizer & tokenizer_;
};

}  // namespace engine::models::qwen3_forced_aligner

#pragma once

#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/tokenizer_text.h"
#include "engine/models/omnivoice/types.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::omnivoice {

class OmniVoicePromptBuilder {
public:
    OmniVoicePromptBuilder(std::shared_ptr<const OmniVoiceAssets> assets, const OmniVoiceTextTokenizer & tokenizer);

    OmniVoicePrompt build(const OmniVoiceRequest & request) const;
    int64_t frame_rate() const;
    int64_t estimate_target_tokens(
        std::string_view text,
        const std::optional<std::string> & ref_text,
        const std::optional<int64_t> & ref_audio_tokens,
        float speed) const;
    std::vector<std::string> chunk_text_punctuation(
        std::string_view text,
        int64_t chunk_len,
        std::optional<int64_t> min_chunk_len = 3) const;

private:
    std::shared_ptr<const OmniVoiceAssets> assets_;
    const OmniVoiceTextTokenizer & tokenizer_;
};

}  // namespace engine::models::omnivoice

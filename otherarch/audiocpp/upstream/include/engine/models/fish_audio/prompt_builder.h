#pragma once

#include "engine/models/fish_audio/tokenizer_text.h"
#include "engine/models/fish_audio/types.h"

namespace engine::models::fish_audio {

class FishAudioPromptBuilder {
public:
    FishAudioPromptBuilder(std::shared_ptr<const FishAudioAssets> assets, FishAudioTextTokenizer tokenizer);

    FishAudioPrompt build(
        const FishAudioRequest & request,
        const std::vector<engine::codecs::FishDacCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn) const;

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    FishAudioTextTokenizer tokenizer_;
};

}  // namespace engine::models::fish_audio

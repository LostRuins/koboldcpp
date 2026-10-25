#pragma once

#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/models/fish_audio/ar.h"
#include "engine/models/fish_audio/prompt_builder.h"
#include "engine/models/fish_audio/tokenizer_text.h"

#include <memory>
#include <functional>
#include <optional>

namespace engine::models::fish_audio {

struct FishAudioGenerationResult {
    runtime::AudioBuffer audio;
    engine::codecs::FishDacCodes codes;
};

class FishAudioGenerator {
public:
    FishAudioGenerator(
        std::shared_ptr<const FishAudioAssets> assets,
        std::unique_ptr<FishAudioDualARRuntime> ar,
        std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec);
    ~FishAudioGenerator();

    engine::codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    FishAudioGenerationResult generate(
        const FishAudioRequest & request,
        const std::vector<engine::codecs::FishDacCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn,
        bool mem_saver,
        bool streaming = false,
        const std::function<void(const runtime::AudioBuffer &)> & on_audio = {});

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    FishAudioTextTokenizer tokenizer_;
    FishAudioPromptBuilder prompt_builder_;
    std::unique_ptr<FishAudioDualARRuntime> ar_;
    std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec_;
};

}  // namespace engine::models::fish_audio

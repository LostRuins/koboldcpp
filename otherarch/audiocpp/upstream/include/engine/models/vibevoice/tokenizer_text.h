#pragma once

#include "engine/models/vibevoice/assets.h"
#include "engine/models/vibevoice/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoicePromptLine {
    int speaker_id = 0;
    std::string text;
};

struct VibeVoicePromptEncoding {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<uint8_t> speech_input_mask;
    std::vector<VibeVoicePromptLine> parsed_script;
    std::vector<int> speaker_ids;
    std::vector<int64_t> speech_prompt_token_counts;
};

class VibeVoiceTextTokenizer {
public:
    struct Impl;

    explicit VibeVoiceTextTokenizer(std::shared_ptr<const VibeVoiceAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    VibeVoicePromptEncoding encode_prompt(const VibeVoiceRequest & request) const;

    int32_t eos_id() const noexcept;
    int32_t pad_id() const noexcept;
    int32_t speech_start_id() const noexcept;
    int32_t speech_end_id() const noexcept;
    int32_t speech_diffusion_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::vibevoice

#pragma once

#include "engine/models/breeze_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::breeze_tts {

struct BreezePromptBranch {
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> text_mask;
    std::vector<int64_t> text_segment_lengths;
    std::vector<std::vector<int32_t>> text_segments;
};

class BreezeTextTokenizer {
public:
    explicit BreezeTextTokenizer(std::shared_ptr<const BreezeTTSAssets> assets);
    ~BreezeTextTokenizer();

    BreezePromptBranch build_tts_instruction(const std::string & text, const std::string & instruction) const;
    BreezePromptBranch build_tts_plain(const std::string & text) const;
    BreezePromptBranch build_clone(
        const std::string & text,
        const std::string & instruction,
        const std::string & reference_text,
        int64_t reference_audio_frames) const;
    BreezePromptBranch build_clone_negative(
        const std::string & text,
        const std::string & reference_text,
        int64_t reference_audio_frames) const;

    int32_t audio_token_id() const noexcept;
    int32_t audio_eos_token_id() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::breeze_tts

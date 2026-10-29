#pragma once

#include "engine/models/neutts/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::neutts {

struct NeuTTSPrompt {
    std::string speaker;
    std::string emotion;
    std::string normalized_reference_text;
    std::string normalized_input_text;
    std::vector<int32_t> token_ids;
    int32_t speech_token_start = 0;
    int32_t speech_token_end = 0;
    int32_t speech_generation_end = 0;
};

class NeuTTSPromptBuilder final {
public:
    explicit NeuTTSPromptBuilder(std::shared_ptr<const NeuTTSAssets> assets);

    NeuTTSPrompt build(
        const std::string & text,
        const std::string & speaker,
        const std::string & emotion) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

std::string normalize_neutts_text(std::string text);

}  // namespace engine::models::neutts

#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::ming3_tts {

struct MingPromptOptions {
    std::string template_name = "tts";
    std::string language;
    std::string instruct;
    std::string instruction_schema;
    std::map<std::string, std::string> controls;
};

struct MingPrompt {
    std::vector<int32_t> ids;
    std::vector<size_t> speaker_positions;
    int64_t latent_position = -1;
};

class MingPromptBuilder {
public:
    explicit MingPromptBuilder(tokenizers::LlamaBpeTokenizerSpec tokenizer);

    MingPrompt build(const std::string & text, const MingPromptOptions & options,
        size_t speakers = 0, const std::string & reference_text = {}, size_t latent_steps = 0) const;

private:
    tokenizers::LlamaBpeTokenizer tokenizer_;
};

}  // namespace engine::models::ming3_tts

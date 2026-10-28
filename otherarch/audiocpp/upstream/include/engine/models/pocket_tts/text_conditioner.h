#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSHostWeights;

struct TextConditionerConfig {
    int64_t hidden_size = 1024;
};

struct TextConditioningResult {
    std::string prepared_text;
    std::vector<int32_t> tokens;
    std::vector<float> text_embeddings;
};

class TextConditioner {
public:
    explicit TextConditioner(TextConditionerConfig config = {});

    TextConditioningResult prepare(
        const PocketTTSAssets & manifest,
        const PocketTTSHostWeights & weights,
        const std::string & text) const;

    // Splits a prompt the way the reference implementation does (pocket_tts
    // split_into_best_sentences): sentence boundaries found on the model's own
    // end-of-sentence tokens, sentences longer than max_tokens split again on
    // ",;:", then regrouped into chunks of at most max_tokens tokens.
    std::vector<std::string> split_into_sentence_chunks(
        const PocketTTSAssets & manifest,
        const std::string & text,
        int64_t max_tokens) const;

private:
    TextConditionerConfig config_;
};

}  // namespace engine::models::pocket_tts

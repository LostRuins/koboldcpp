#ifndef KCPP_SDCPP_GGUF_TOKENIZER_H
#define KCPP_SDCPP_GGUF_TOKENIZER_H

#include <string>

namespace kcpp_sd {

enum class tokenizer_profile {
    DEFAULT,
    MOSS_IMAGE,
};

// Converts tokenizer metadata embedded in a text-encoder GGUF to a Hugging
// Face tokenizer.json document without writing an intermediate file.
class gguf_tokenizer_extractor {
public:
    static bool to_hf_json(const std::string& gguf_path,
                           std::string& json,
                           std::string& error,
                           tokenizer_profile profile = tokenizer_profile::DEFAULT);

    // Builds the subset of tokenizer.json consumed by T5UniGramTokenizer.
    static bool to_unigram_json(const std::string& gguf_path,
                                std::string& json,
                                std::string& error);

    // Builds the two legacy resources consumed by GemmaTokenizer.
    static bool to_vocab_and_merges(const std::string& gguf_path,
                                    std::string& vocab_json,
                                    std::string& merges,
                                    std::string& error);
};

}  // namespace kcpp_sd

#endif  // KCPP_SDCPP_GGUF_TOKENIZER_H

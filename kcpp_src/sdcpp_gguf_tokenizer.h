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
};

}  // namespace kcpp_sd

#endif  // KCPP_SDCPP_GGUF_TOKENIZER_H

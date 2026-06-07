#ifndef __SD_TOKENIZERS_MISTRAL_TOKENIZER_H__
#define __SD_TOKENIZERS_MISTRAL_TOKENIZER_H__

#include <string>

#include "bpe_tokenizer.h"

class MistralTokenizer : public BPETokenizer {
protected:
    void load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str);

public:
    explicit MistralTokenizer(const std::string& merges_utf8_str = "", const std::string& vocab_utf8_str = "");
};

#endif  // __SD_TOKENIZERS_MISTRAL_TOKENIZER_H__

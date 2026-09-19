#ifndef __SD_TOKENIZERS_GEMMA_TOKENIZER_H__
#define __SD_TOKENIZERS_GEMMA_TOKENIZER_H__

#include <string>

#include "bpe_tokenizer.h"

class GemmaTokenizer : public BPETokenizer {
protected:
    void load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str);
    std::string normalize(const std::string& text) const override;

public:
    explicit GemmaTokenizer(const std::string& merges_utf8_str = "", const std::string& vocab_utf8_str = "");
};

#endif  // __SD_TOKENIZERS_GEMMA_TOKENIZER_H__

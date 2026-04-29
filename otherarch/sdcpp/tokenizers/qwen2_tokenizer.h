#ifndef __SD_TOKENIZERS_QWEN2_TOKENIZER_H__
#define __SD_TOKENIZERS_QWEN2_TOKENIZER_H__

#include <string>

#include "bpe_tokenizer.h"

class Qwen2Tokenizer : public BPETokenizer {
protected:
    void load_from_merges(const std::string& merges_utf8_str);

public:
    explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "");
};

#endif  // __SD_TOKENIZERS_QWEN2_TOKENIZER_H__

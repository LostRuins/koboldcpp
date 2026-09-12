#ifndef __SD_TOKENIZERS_SENSENOVA_U1_TOKENIZER_H__
#define __SD_TOKENIZERS_SENSENOVA_U1_TOKENIZER_H__

#include <string>

#include "qwen2_tokenizer.h"

class SenseNovaU1Tokenizer : public Qwen2Tokenizer {
public:
    explicit SenseNovaU1Tokenizer(const std::string& merges_utf8_str = "");
};

#endif  // __SD_TOKENIZERS_SENSENOVA_U1_TOKENIZER_H__

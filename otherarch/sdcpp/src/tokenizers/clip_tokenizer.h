#ifndef __SD_TOKENIZERS_CLIP_TOKENIZER_H__
#define __SD_TOKENIZERS_CLIP_TOKENIZER_H__

#include <cstddef>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"

class CLIPTokenizer : public BPETokenizer {
protected:
    void load_from_merges(const std::string& merges_utf8_str);
    std::string normalize(const std::string& text) const override;
    std::vector<std::string> token_split(const std::string& text) const override;

public:
    explicit CLIPTokenizer(int pad_token_id = 49407, const std::string& merges_utf8_str = "");
};

#endif  // __SD_TOKENIZERS_CLIP_TOKENIZER_H__

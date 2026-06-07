#ifndef __SD_TOKENIZERS_BPE_TOKENIZER_H__
#define __SD_TOKENIZERS_BPE_TOKENIZER_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tokenizer.h"

class BPETokenizer : public Tokenizer {
protected:
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> byte_decoder;
    std::map<std::u32string, int> encoder;
    std::map<int, std::u32string> decoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    int encoder_len     = 0;
    int bpe_len         = 0;
    bool byte_level_bpe = true;
    bool byte_fallback  = false;

protected:
    static std::vector<std::pair<int, std::u32string>> bytes_to_unicode();
    static std::vector<std::u32string> split_utf32(const std::string& text, char32_t delimiter = U'\n');
    virtual std::vector<std::string> token_split(const std::string& text) const;
    std::vector<std::u32string> bpe(const std::u32string& token) const;
    std::string decode_token(int token_id) const override;

public:
    BPETokenizer()          = default;
    virtual ~BPETokenizer() = default;

    std::vector<int> encode(const std::string& text, on_new_token_cb_t on_new_token_cb = nullptr) override;
};

#endif  // __SD_TOKENIZERS_BPE_TOKENIZER_H__

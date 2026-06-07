#ifndef __SD_TOKENIZERS_TOKENIZER_H__
#define __SD_TOKENIZERS_TOKENIZER_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using on_new_token_cb_t = std::function<bool(std::string&, std::vector<int32_t>&)>;

class Tokenizer {
protected:
    std::vector<std::string> special_tokens;
    bool add_bos_token = false;
    bool add_eos_token = false;
    bool pad_left      = false;
    std::string end_of_word_suffix;

    virtual std::string decode_token(int token_id) const = 0;
    virtual std::string normalize(const std::string& text) const;

public:
    std::string UNK_TOKEN;
    std::string BOS_TOKEN;
    std::string EOS_TOKEN;
    std::string PAD_TOKEN;
    int UNK_TOKEN_ID = 0;
    int BOS_TOKEN_ID = 0;
    int EOS_TOKEN_ID = 0;
    int PAD_TOKEN_ID = 0;

    virtual ~Tokenizer() = default;

    void add_special_token(const std::string& token);
    bool is_special_token(const std::string& token) const;
    virtual std::vector<int> encode(const std::string& text, on_new_token_cb_t on_new_token_cb = nullptr) = 0;
    std::vector<int> tokenize(const std::string& text,
                              on_new_token_cb_t on_new_token_cb = nullptr,
                              bool padding                      = false,
                              size_t min_length                 = 0,
                              size_t max_length                 = 100000000,
                              bool allow_overflow_expand        = false);
    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>* weights,
                    std::vector<float>* mask,
                    size_t min_length          = 0,
                    size_t max_length          = 100000000,
                    bool allow_overflow_expand = false);
    std::string decode(const std::vector<int>& tokens) const;
};

#endif  // __SD_TOKENIZERS_TOKENIZER_H__

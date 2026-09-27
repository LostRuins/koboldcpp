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
    bool add_bos_token          = false;
    bool add_eos_token          = false;
    bool pad_left               = false;
    bool normalize_before_split = false;
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

    virtual void add_special_token(const std::string& token);
    bool is_special_token(const std::string& token) const;
    // An empty output may be valid; failures return false and clear the output.
    virtual bool encode(const std::string& text, std::vector<int>& tokens, on_new_token_cb_t on_new_token_cb = nullptr, std::string* error = nullptr) = 0;
    bool tokenize(const std::string& text,
                  std::vector<int>& tokens,
                  on_new_token_cb_t on_new_token_cb = nullptr,
                  bool padding                      = false,
                  size_t min_length                 = 0,
                  size_t max_length                 = 100000000,
                  bool allow_overflow_expand        = false,
                  std::string* error                = nullptr);
    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>* weights,
                    std::vector<float>* mask,
                    size_t min_length          = 0,
                    size_t max_length          = 100000000,
                    bool allow_overflow_expand = false);
    virtual bool decode(const std::vector<int>& tokens, std::string& text, std::string* error = nullptr) const;
};

#endif  // __SD_TOKENIZERS_TOKENIZER_H__

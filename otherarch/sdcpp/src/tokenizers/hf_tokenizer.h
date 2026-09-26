#ifndef __SD_TOKENIZERS_HF_TOKENIZER_H__
#define __SD_TOKENIZERS_HF_TOKENIZER_H__

#include <memory>

#include "tokenizer.h"

class HFTokenizer : public Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string decode_token(int token_id) const override;

public:
    explicit HFTokenizer(const std::string& path);
    ~HFTokenizer() override;

    // Padding belongs to the encoder; tokenizer.json supplies the single-sequence template.
    void set_padding(int token_id, bool left);
    void validate_vocab_size(int64_t embedding_rows) const;
    int token_to_id(const std::string& token) const;
    void add_special_token(const std::string& token) override;
    bool encode(const std::string& text, std::vector<int>& tokens, on_new_token_cb_t on_new_token_cb = nullptr, std::string* error = nullptr) override;
    bool decode(const std::vector<int>& tokens, std::string& text, std::string* error = nullptr) const override;
};

#endif  // __SD_TOKENIZERS_HF_TOKENIZER_H__

#ifndef __SD_TOKENIZERS_TOKENIZER_CONFIG_H__
#define __SD_TOKENIZERS_TOKENIZER_CONFIG_H__

#include <array>
#include <memory>
#include <string>

#include "tokenizer.h"

class TokenizerConfig {
public:
    enum Slot { MAIN,
                CLIP_L,
                CLIP_G };

private:
    std::array<std::string, 3> paths_;
    mutable std::array<bool, 3> used_{};

public:
    TokenizerConfig() = default;
    explicit TokenizerConfig(const char* config);
    bool has(Slot slot) const;
    std::shared_ptr<Tokenizer> create(Slot slot, int64_t embedding_rows, int padding_id, bool pad_left = false, bool clip = false) const;
    void validate_usage() const;
};

#endif  // __SD_TOKENIZERS_TOKENIZER_CONFIG_H__

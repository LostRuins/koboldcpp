#pragma once

#include "gguf.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace qwen3_tts {

// BPE tokenizer configuration
struct tokenizer_config {
    int32_t vocab_size = 151936;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;  // <|im_end|>
    int32_t bos_token_id = 151644;  // <|im_start|>
};

// Text tokenizer class (BPE-based, GPT-2 style with Qwen2 pre-tokenization)
class TextTokenizer {
public:
    TextTokenizer();
    ~TextTokenizer();

    // Load tokenizer from GGUF file
    bool load_from_gguf(struct gguf_context * ctx);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string & text) const;

    // Encode with TTS format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int32_t> encode_for_tts(const std::string & text) const;

    std::vector<int32_t> encode_instruct(const std::string & instruct) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t> & tokens) const;

    // Decode single token
    std::string decode_token(int32_t token_id) const;

    // Get configuration
    const tokenizer_config & get_config() const { return config_; }

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if loaded
    bool is_loaded() const { return loaded_; }

    // Get special token IDs
    int32_t bos_token_id() const { return config_.bos_token_id; }
    int32_t eos_token_id() const { return config_.eos_token_id; }
    int32_t pad_token_id() const { return config_.pad_token_id; }

private:
    tokenizer_config config_;
    std::string error_msg_;
    bool loaded_ = false;

    // Vocabulary: token string -> token ID
    std::unordered_map<std::string, int32_t> vocab_;

    // Reverse vocabulary: token ID -> token string
    std::vector<std::string> id_to_token_;

    // BPE merges: pair -> rank (lower rank = higher priority)
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;

    // Special token for "assistant" and newline
    int32_t assistant_token_id_ = 77091;
    int32_t newline_token_id_ = 198;  // '\n' encoded

    // Helper: convert bytes to unicode (GPT-2 style byte encoding)
    static std::string bytes_to_unicode(const std::string & text);
    static std::string unicode_to_bytes(const std::string & text);

    // Helper: get UTF-8 character length
    static size_t utf8_len(char c);

    // BPE encoding for a single word
    std::vector<std::string> bpe(const std::string & token) const;

    // Find the pair with lowest rank in a sequence
    std::pair<std::string, std::string> get_min_pair(
        const std::vector<std::string> & word) const;
};

} // namespace qwen3_tts

#include "text_tokenizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace qwen3_tts {

// GPT-2 byte-to-unicode mapping
// Maps bytes 0-255 to unicode characters to avoid control characters
static const char * BYTE_TO_UNICODE[256] = {
    "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
    "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
    "Ġ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ġ",
    "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
    "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
    "ł", "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "Ń", "®", "¯",
    "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",
    "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
    "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",
    "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï",
    "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ"
};

// Build reverse mapping at runtime
static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int i = 0; i < 256; i++) {
        result[BYTE_TO_UNICODE[i]] = (uint8_t)i;
    }
    return result;
}

static const std::unordered_map<std::string, uint8_t> UNICODE_TO_BYTE = build_unicode_to_byte();

TextTokenizer::TextTokenizer() = default;

TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

std::string TextTokenizer::bytes_to_unicode(const std::string & text) {
    std::string result;
    for (unsigned char c : text) {
        result += BYTE_TO_UNICODE[c];
    }
    return result;
}

std::string TextTokenizer::unicode_to_bytes(const std::string & text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_len(text[i]);
        std::string ch = text.substr(i, len);
        auto it = UNICODE_TO_BYTE.find(ch);
        if (it != UNICODE_TO_BYTE.end()) {
            result += (char)it->second;
        } else {
            // Not in mapping, keep as-is (shouldn't happen for valid tokens)
            result += ch;
        }
        i += len;
    }
    return result;
}

bool TextTokenizer::load_from_gguf(struct gguf_context * ctx) {
    if (!ctx) {
        error_msg_ = "GGUF context is null";
        return false;
    }

    // Get vocabulary
    int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        error_msg_ = "tokenizer.ggml.tokens not found in GGUF";
        return false;
    }

    size_t n_vocab = gguf_get_arr_n(ctx, tokens_key);
    if (n_vocab == 0) {
        error_msg_ = "Empty vocabulary";
        return false;
    }

    config_.vocab_size = (int32_t)n_vocab;
    id_to_token_.resize(n_vocab);

    for (size_t i = 0; i < n_vocab; i++) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            id_to_token_[i] = token;
            vocab_[token] = (int32_t)i;
        }
    }

    // Get merges
    int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, merges_key);
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(ctx, merges_key, i);
            if (merge) {
                std::string merge_str(merge);
                // Parse "token1 token2" format
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    bpe_ranks_[{first, second}] = (int32_t)i;
                }
            }
        }
    }

    // Get special token IDs (optional, use defaults if not found)
    int64_t bos_key = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) {
        config_.bos_token_id = (int32_t)gguf_get_val_u32(ctx, bos_key);
    }

    int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        config_.eos_token_id = (int32_t)gguf_get_val_u32(ctx, eos_key);
    }

    int64_t pad_key = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key >= 0) {
        config_.pad_token_id = (int32_t)gguf_get_val_u32(ctx, pad_key);
    }

    // Find special tokens by content
    auto find_token = [this](const std::string & text) -> int32_t {
        auto it = vocab_.find(text);
        return (it != vocab_.end()) ? it->second : -1;
    };

    assistant_token_id_ = find_token("assistant");
    if (assistant_token_id_ < 0) {
        // Try with space prefix (GPT-2 style)
        assistant_token_id_ = find_token("Ġassistant");
    }

    // Newline token
    newline_token_id_ = find_token("Ċ");  // GPT-2 encoding for '\n'
    if (newline_token_id_ < 0) {
        newline_token_id_ = find_token("\n");
    }

    loaded_ = true;
    return true;
}

std::pair<std::string, std::string> TextTokenizer::get_min_pair(
    const std::vector<std::string> & word) const {

    std::pair<std::string, std::string> min_pair;
    int32_t min_rank = std::numeric_limits<int32_t>::max();

    for (size_t i = 0; i + 1 < word.size(); i++) {
        auto pair = std::make_pair(word[i], word[i + 1]);
        auto it = bpe_ranks_.find(pair);
        if (it != bpe_ranks_.end() && it->second < min_rank) {
            min_rank = it->second;
            min_pair = pair;
        }
    }

    return min_pair;
}

std::vector<std::string> TextTokenizer::bpe(const std::string & token) const {
    if (token.empty()) {
        return {};
    }

    // Split into unicode characters
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        size_t len = utf8_len(token[i]);
        word.push_back(token.substr(i, len));
        i += len;
    }

    if (word.size() == 1) {
        return word;
    }

    // Iteratively merge pairs
    while (true) {
        auto min_pair = get_min_pair(word);
        if (min_pair.first.empty()) {
            break;  // No more merges possible
        }

        // Merge all occurrences of the pair
        std::vector<std::string> new_word;
        size_t j = 0;
        while (j < word.size()) {
            if (j + 1 < word.size() &&
                word[j] == min_pair.first &&
                word[j + 1] == min_pair.second) {
                new_word.push_back(min_pair.first + min_pair.second);
                j += 2;
            } else {
                new_word.push_back(word[j]);
                j += 1;
            }
        }
        word = std::move(new_word);

        if (word.size() == 1) {
            break;
        }
    }

    return word;
}

std::vector<int32_t> TextTokenizer::encode(const std::string & text) const {
    if (!loaded_) {
        return {};
    }

    std::vector<int32_t> tokens;

    // Convert text to GPT-2 unicode representation
    std::string unicode_text = bytes_to_unicode(text);

    // Simple word splitting (no regex pre-tokenization for now)
    // Split on spaces but keep the space with the following word (GPT-2 style)
    std::vector<std::string> words;
    std::string current_word;

    size_t i = 0;
    while (i < unicode_text.size()) {
        size_t len = utf8_len(unicode_text[i]);
        std::string ch = unicode_text.substr(i, len);

        // Check if this is a space (Ġ in GPT-2 encoding)
        if (ch == "Ġ") {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            current_word = ch;  // Start new word with space
        } else {
            current_word += ch;
        }
        i += len;
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    // BPE encode each word
    for (const auto & word : words) {
        auto bpe_tokens = bpe(word);
        for (const auto & tok : bpe_tokens) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // Unknown token - encode as bytes
                for (unsigned char c : tok) {
                    std::string byte_tok = BYTE_TO_UNICODE[c];
                    auto byte_it = vocab_.find(byte_tok);
                    if (byte_it != vocab_.end()) {
                        tokens.push_back(byte_it->second);
                    }
                }
            }
        }
    }

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_instruct(const std::string & instruct) const {
    if (!loaded_ || instruct.empty()) {
        return {};
    }

    // Format: <|im_start|>user\n{instruct}<|im_end|>\n
    std::vector<int32_t> tokens;

    // <|im_start|>
    tokens.push_back(config_.bos_token_id);

    // user
    int user_token_id_ = 872;
    tokens.push_back(user_token_id_);

    // \n
    tokens.push_back(newline_token_id_);

    // Encode the instruct
    auto text_tokens = encode(instruct);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    // <|im_end|>
    tokens.push_back(config_.eos_token_id);

    // \n
    tokens.push_back(newline_token_id_);

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_for_tts(const std::string & text) const {
    if (!loaded_) {
        return {};
    }

    // Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int32_t> tokens;

    // <|im_start|>
    tokens.push_back(config_.bos_token_id);

    // assistant
    tokens.push_back(assistant_token_id_);

    // \n
    tokens.push_back(newline_token_id_);

    // Encode the text
    auto text_tokens = encode(text);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    // <|im_end|>
    tokens.push_back(config_.eos_token_id);

    // \n
    tokens.push_back(newline_token_id_);

    // <|im_start|>
    tokens.push_back(config_.bos_token_id);

    // assistant
    tokens.push_back(assistant_token_id_);

    // \n
    tokens.push_back(newline_token_id_);

    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)id_to_token_.size()) {
        return "";
    }

    const std::string & token = id_to_token_[token_id];

    // Convert from GPT-2 unicode back to bytes
    return unicode_to_bytes(token);
}

} // namespace qwen3_tts
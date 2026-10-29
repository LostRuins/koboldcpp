#include "engine/models/yue2/tokenizer_text.h"

#include "engine/models/yue2/types.h"

#include "bpe-core.h"
#include "unicode.h"

#include <array>
#include <climits>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine::models::yue2 {
namespace {

namespace vendor = llama_tokenizer_vendor;

std::string decode_base64(const std::string & input) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> values{};
        values.fill(-1);
        for (int i = 0; i < 26; ++i) {
            values[static_cast<size_t>('A' + i)] = static_cast<int8_t>(i);
            values[static_cast<size_t>('a' + i)] = static_cast<int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            values[static_cast<size_t>('0' + i)] = static_cast<int8_t>(52 + i);
        }
        values[static_cast<size_t>('+')] = 62;
        values[static_cast<size_t>('/')] = 63;
        return values;
    }();
    std::string out;
    int bits = 0;
    int value = 0;
    for (const unsigned char ch : input) {
        if (ch == '=') {
            break;
        }
        const int8_t digit = table[ch];
        if (digit < 0) {
            throw std::runtime_error("Yue2 tiktoken vocabulary contains invalid base64 bytes");
        }
        value = (value << 6) | digit;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((value >> bits) & 0xff));
        }
    }
    return out;
}

std::string map_token_bytes(const std::string & bytes) {
    std::string mapped;
    for (const unsigned char byte : bytes) {
        mapped += unicode_byte_to_utf8(byte);
    }
    return mapped;
}

std::string pair_key(const std::string & left, const std::string & right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

void add_special_token(vendor::BpeVocabulary & vocab, const std::string & text, int32_t id) {
    vocab.token_to_id.emplace(text, id);
    vocab.id_to_token.emplace(id, vendor::TokenData{text, vendor::TOKEN_ATTR_CONTROL});
}

void register_yue2_special_tokens(vendor::BpeVocabulary & vocab, int32_t base_id) {
    static const std::array<const char *, 8> kBaseSpecials = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<R>",
        "<S>",
        "<X>",
        "<mask>",
        "<sep>",
    };
    int32_t id = base_id;
    for (const char * token : kBaseSpecials) {
        add_special_token(vocab, token, id++);
    }
    for (int i = 0; i < 200; ++i) {
        add_special_token(vocab, "<extra_" + std::to_string(i) + ">", id++);
    }
    add_special_token(vocab, "<abc>", kAbcStartToken);
    add_special_token(vocab, "</abc>", kAbcEndToken);
    add_special_token(vocab, "<music>", kMusicStartToken);
    add_special_token(vocab, "</music>", kMusicEndToken);
}

std::shared_ptr<vendor::BpeVocabulary> load_tiktoken_vocabulary(const std::filesystem::path & vocab_path) {
    std::ifstream input(vocab_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open Yue2 tiktoken vocabulary: " + vocab_path.string());
    }
    auto vocab = std::make_shared<vendor::BpeVocabulary>();
    vocab->pre_type = vendor::PreTokenizerType::Qwen2;

    std::string line;
    int64_t mergeable_count = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        std::istringstream parts(line);
        std::string token_base64;
        int64_t rank = -1;
        if (!(parts >> token_base64 >> rank) || rank < 0 || rank > INT_MAX) {
            throw std::runtime_error("Yue2 tiktoken vocabulary has an invalid line: " + line);
        }
        const std::string bytes = decode_base64(token_base64);
        const auto token_id = static_cast<int32_t>(rank);
        const std::string mapped = map_token_bytes(bytes);
        vocab->token_to_id.emplace(mapped, token_id);
        vocab->id_to_token.emplace(token_id, vendor::TokenData{mapped, 0});
        for (size_t split = 1; split < bytes.size(); ++split) {
            vocab->bpe_ranks.emplace(
                pair_key(map_token_bytes(bytes.substr(0, split)), map_token_bytes(bytes.substr(split))),
                token_id);
        }
        ++mergeable_count;
    }
    if (mergeable_count != kEodToken) {
        throw std::runtime_error("Yue2 tiktoken mergeable rank count must be 151643");
    }
    register_yue2_special_tokens(*vocab, static_cast<int32_t>(mergeable_count));
    vendor::rebuild_special_tokens_cache(*vocab);
    return vocab;
}

}  // namespace

Yue2TextTokenizer::Yue2TextTokenizer(const std::filesystem::path & vocab_path)
    : vocab_(load_tiktoken_vocabulary(vocab_path)) {}

std::vector<int32_t> Yue2TextTokenizer::encode(const std::string & text) const {
    return vendor::tokenize_bpe(*vocab_, text, true);
}

std::string Yue2TextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    return vendor::decode_bpe(*vocab_, token_ids, false);
}

}  // namespace engine::models::yue2

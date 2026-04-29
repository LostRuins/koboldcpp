#include "bpe_tokenizer.h"

#include <algorithm>
#include <sstream>

#include "tokenize_util.h"
#include "util.h"

std::vector<std::pair<int, std::u32string>> BPETokenizer::bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    return byte_unicode_pairs;
}

std::vector<std::string> BPETokenizer::token_split(const std::string& text) const {
    return ::token_split(text);
}

std::vector<std::u32string> BPETokenizer::split_utf32(const std::string& text, char32_t delimiter) {
    std::vector<std::u32string> result;
    size_t start              = 0;
    size_t pos                = 0;
    std::u32string utf32_text = utf8_to_utf32(text);
    while ((pos = utf32_text.find(delimiter, start)) != std::u32string::npos) {
        result.push_back(utf32_text.substr(start, pos - start));
        start = pos + 1;
    }
    return result;
}

static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
    std::set<std::pair<std::u32string, std::u32string>> pairs;
    if (subwords.empty()) {
        return pairs;
    }

    std::u32string prev_subword = subwords[0];
    for (int i = 1; i < static_cast<int>(subwords.size()); i++) {
        std::u32string subword = subwords[i];
        std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
        pairs.insert(pair);
        prev_subword = subword;
    }
    return pairs;
}

std::vector<std::u32string> BPETokenizer::bpe(const std::u32string& token) const {
    std::vector<std::u32string> word;

    for (int i = 0; i < static_cast<int>(token.size()) - 1; i++) {
        word.emplace_back(1, token[i]);
    }
    word.push_back(token.substr(token.size() - 1) + utf8_to_utf32(end_of_word_suffix));

    std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

    if (pairs.empty()) {
        return {token + utf8_to_utf32(end_of_word_suffix)};
    }

    while (true) {
        auto min_pair_iter = std::min_element(pairs.begin(),
                                              pairs.end(),
                                              [&](const std::pair<std::u32string, std::u32string>& a,
                                                  const std::pair<std::u32string, std::u32string>& b) {
                                                  if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                      return false;
                                                  } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                      return true;
                                                  }
                                                  return bpe_ranks.at(a) < bpe_ranks.at(b);
                                              });

        const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }

        std::u32string first  = bigram.first;
        std::u32string second = bigram.second;
        std::vector<std::u32string> new_word;
        int32_t i = 0;

        while (i < static_cast<int32_t>(word.size())) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = static_cast<int32_t>(std::distance(word.begin(), it));

            if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;

        if (word.size() == 1) {
            break;
        }
        pairs = get_pairs(word);
    }

    return word;
}

std::vector<int> BPETokenizer::encode(const std::string& text, on_new_token_cb_t on_new_token_cb) {
    std::string normalized_text = normalize(text);
    std::vector<int32_t> bpe_tokens;
    std::vector<std::string> token_strs;

    auto splited_texts = split_with_special_tokens(normalized_text, special_tokens);

    for (auto& splited_text : splited_texts) {
        if (is_special_token(splited_text)) {
            if (on_new_token_cb != nullptr) {
                bool skip = on_new_token_cb(splited_text, bpe_tokens);
                if (skip) {
                    token_strs.push_back(splited_text);
                    continue;
                }
            }
            bpe_tokens.push_back(encoder[utf8_to_utf32(splited_text)]);
            token_strs.push_back(splited_text);
            continue;
        }
        auto tokens = token_split(splited_text);
        for (auto& token : tokens) {
            if (on_new_token_cb != nullptr) {
                bool skip = on_new_token_cb(token, bpe_tokens);
                if (skip) {
                    token_strs.push_back(splited_text);
                    continue;
                }
            }

            std::string token_str = token;
            std::u32string utf32_token;
            for (int i = 0; i < static_cast<int>(token_str.length()); i++) {
                unsigned char b = token_str[i];
                utf32_token += byte_encoder[b];
            }
            auto bpe_strs = bpe(utf32_token);
            for (auto bpe_str : bpe_strs) {
                bpe_tokens.push_back(encoder[bpe_str]);
                token_strs.push_back(utf32_to_utf8(bpe_str));
            }
        }
    }

    std::stringstream ss;
    ss << "[";
    for (auto token : token_strs) {
        ss << "\"" << token << "\", ";
    }
    ss << "]";
    LOG_DEBUG("split prompt \"%s\" to tokens %s", text.c_str(), ss.str().c_str());
    return bpe_tokens;
}

std::string BPETokenizer::decode_token(int token_id) const {
    return utf32_to_utf8(decoder.at(token_id));
}

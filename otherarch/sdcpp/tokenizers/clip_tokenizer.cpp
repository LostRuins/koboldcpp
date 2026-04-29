#include "clip_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <set>

#include "ggml.h"
#include "tokenize_util.h"
#include "util.h"
#include "vocab/vocab.h"

CLIPTokenizer::CLIPTokenizer(int pad_token_id, const std::string& merges_utf8_str) {
    UNK_TOKEN = "<|endoftext|>";
    BOS_TOKEN = "<|startoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    UNK_TOKEN_ID = 49407;
    BOS_TOKEN_ID = 49406;
    EOS_TOKEN_ID = 49407;
    PAD_TOKEN_ID = pad_token_id;

    end_of_word_suffix = "</w>";
    add_bos_token      = true;
    add_eos_token      = true;

    if (merges_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str);
    } else {
        load_from_merges(load_clip_merges());
    }
    add_special_token("<|startoftext|>");
    add_special_token("<|endoftext|>");
}

void CLIPTokenizer::load_from_merges(const std::string& merges_utf8_str) {
    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    GGML_ASSERT(merges.size() == 48895);
    merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }
    std::vector<std::u32string> vocab;
    for (const auto& pair : byte_unicode_pairs) {
        vocab.push_back(pair.second);
    }
    for (const auto& pair : byte_unicode_pairs) {
        vocab.push_back(pair.second + utf8_to_utf32("</w>"));
    }
    for (const auto& merge : merge_pairs) {
        vocab.push_back(merge.first + merge.second);
    }
    vocab.push_back(utf8_to_utf32("<|startoftext|>"));
    vocab.push_back(utf8_to_utf32("<|endoftext|>"));
    LOG_DEBUG("vocab size: %llu", vocab.size());
    int i = 0;
    for (const auto& token : vocab) {
        encoder[token] = i;
        decoder[i]     = token;
        i++;
    }
    encoder_len = i;

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

static std::string strip(const std::string& str) {
    std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
    std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

    if (start == std::string::npos) {
        return "";
    }

    return str.substr(start, end - start + 1);
}

static std::string whitespace_clean(const std::string& text) {
    auto result = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    result      = strip(result);
    return result;
}

std::string CLIPTokenizer::normalize(const std::string& text) const {
    auto normalized_text = whitespace_clean(text);
    std::transform(normalized_text.begin(), normalized_text.end(), normalized_text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized_text;
}

std::vector<std::string> CLIPTokenizer::token_split(const std::string& text) const {
    std::regex clip_pat(R"('s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                        std::regex::icase);
    std::sregex_iterator iter(text.begin(), text.end(), clip_pat);
    std::sregex_iterator end;

    std::vector<std::string> result;
    for (; iter != end; ++iter) {
        result.emplace_back(iter->str());
    }

    return result;
}

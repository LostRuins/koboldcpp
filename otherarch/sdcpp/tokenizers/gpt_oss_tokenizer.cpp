#include "gpt_oss_tokenizer.h"

#include "json.hpp"
#include "util.h"
#include "vocab/vocab.h"

void GPTOSSTokenizer::load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }

    nlohmann::json vocab;
    try {
        vocab = nlohmann::json::parse(vocab_utf8_str);
    } catch (const nlohmann::json::parse_error&) {
        GGML_ABORT("invalid vocab json str");
    }
    for (const auto& [key, value] : vocab.items()) {
        std::u32string token = utf8_to_utf32(key);
        int i                = value;
        encoder[token]       = i;
        decoder[i]           = token;
    }
    encoder_len = static_cast<int>(encoder.size());
    for (auto& special_token : special_tokens) {
        auto token           = utf8_to_utf32(special_token);
        encoder[token]       = encoder_len;
        decoder[encoder_len] = token;
        encoder_len++;
    }
    encoder_len = static_cast<int>(encoder.size());
    LOG_DEBUG("vocab size: %d", encoder_len);

    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }
    LOG_DEBUG("merges size %zu", merge_pairs.size());

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

GPTOSSTokenizer::GPTOSSTokenizer(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    BOS_TOKEN = "<|startoftext|>";
    UNK_TOKEN = "<|endoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    BOS_TOKEN_ID = 199998;
    EOS_TOKEN_ID = 199999;
    UNK_TOKEN_ID = 199999;
    PAD_TOKEN_ID = 199999;

    special_tokens = {
        "<|startoftext|>",
        "<|endoftext|>",
        "<|reserved_200000|>",
        "<|reserved_200001|>",
        "<|return|>",
        "<|constrain|>",
        "<|reserved_200004|>",
        "<|channel|>",
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|reserved_200009|>",
        "<|reserved_200010|>",
        "<|reserved_200011|>",
        "<|call|>",
        "<|reserved_200013|>",
        "<|reserved_200014|>",
        "<|reserved_200015|>",
        "<|reserved_200016|>",
        "<|reserved_200017|>",
        "<|endofprompt|>",
    };

    if (merges_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str, vocab_utf8_str);
    } else {
        load_from_merges(load_gpt_oss_merges(), load_gpt_oss_vocab_json());
    }
}

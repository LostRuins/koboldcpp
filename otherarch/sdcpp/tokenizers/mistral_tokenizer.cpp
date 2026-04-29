#include "mistral_tokenizer.h"

#include "ggml.h"
#include "json.hpp"
#include "util.h"
#include "vocab/vocab.h"

void MistralTokenizer::load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
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
    encoder_len = static_cast<int>(vocab.size());
    LOG_DEBUG("vocab size: %d", encoder_len);

    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }
    std::vector<std::u32string> merges = split_utf32(merges_utf8_str);
    LOG_DEBUG("merges size %llu", merges.size());
    std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
    for (const auto& merge : merges) {
        size_t space_pos = merge.find(' ');
        merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
    }

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

MistralTokenizer::MistralTokenizer(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
    add_bos_token = true;

    UNK_TOKEN = "<unk>";
    BOS_TOKEN = "<s>";
    EOS_TOKEN = "</s>";
    PAD_TOKEN = "<pad>";

    UNK_TOKEN_ID = 0;
    BOS_TOKEN_ID = 1;
    EOS_TOKEN_ID = 2;
    PAD_TOKEN_ID = 11;

    special_tokens = {
        "<unk>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
        "[TOOL_CALLS]",
        "[IMG]",
        "<pad>",
        "[IMG_BREAK]",
        "[IMG_END]",
        "[PREFIX]",
        "[MIDDLE]",
        "[SUFFIX]",
        "[SYSTEM_PROMPT]",
        "[/SYSTEM_PROMPT]",
        "[TOOL_CONTENT]",
    };
    for (int i = 20; i < 1000; i++) {
        special_tokens.push_back("<SPECIAL_" + std::to_string(i) + ">");
    }

    if (merges_utf8_str.size() > 0 && vocab_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str, vocab_utf8_str);
    } else {
        load_from_merges(load_mistral_merges(), load_mistral_vocab_json());
    }
}

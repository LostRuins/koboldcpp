#include "qwen2_tokenizer.h"

#include "util.h"
#include "vocab/vocab.h"

void Qwen2Tokenizer::load_from_merges(const std::string& merges_utf8_str) {
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

    std::vector<std::u32string> tokens;
    for (const auto& pair : byte_unicode_pairs) {
        tokens.push_back(pair.second);
    }
    for (const auto& merge : merge_pairs) {
        tokens.push_back(merge.first + merge.second);
    }
    for (auto& special_token : special_tokens) {
        tokens.push_back(utf8_to_utf32(special_token));
    }

    int i = 0;
    for (const auto& token : tokens) {
        encoder[token] = i;
        decoder[i]     = token;
        i++;
    }
    encoder_len = i;
    LOG_DEBUG("vocab size: %d", encoder_len);

    int rank = 0;
    for (const auto& merge : merge_pairs) {
        bpe_ranks[merge] = rank++;
    }
    bpe_len = rank;
}

Qwen2Tokenizer::Qwen2Tokenizer(const std::string& merges_utf8_str) {
    UNK_TOKEN = "<|endoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    UNK_TOKEN_ID = 151643;
    EOS_TOKEN_ID = 151643;
    PAD_TOKEN_ID = 151643;

    special_tokens = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    };

    if (merges_utf8_str.size() > 0) {
        load_from_merges(merges_utf8_str);
    } else {
        load_from_merges(load_qwen2_merges());
    }
}

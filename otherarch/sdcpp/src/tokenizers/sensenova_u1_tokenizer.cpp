#include "sensenova_u1_tokenizer.h"

#include <vector>

static const std::vector<std::string>& sensenova_u1_special_tokens() {
    static const std::vector<std::string> tokens = {
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
        "<IMG_CONTEXT>",
        "<img>",
        "</img>",
    };
    return tokens;
}

SenseNovaU1Tokenizer::SenseNovaU1Tokenizer(const std::string& merges_utf8_str)
    : Qwen2Tokenizer(merges_utf8_str, sensenova_u1_special_tokens()) {
    EOS_TOKEN    = "<|im_end|>";
    EOS_TOKEN_ID = 151645;
}

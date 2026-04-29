#include "vocab.h"
#include "clip_t5.hpp"
#include "mistral.hpp"
#include "qwen.hpp"
#include "umt5.hpp"

std::string load_clip_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_qwen2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_mistral_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(mistral_merges_utf8_c_str), sizeof(mistral_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_mistral_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(mistral_vocab_json_utf8_c_str), sizeof(mistral_vocab_json_utf8_c_str));
    return json_str;
}

std::string load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string load_umt5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
    return json_str;
}
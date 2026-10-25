#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_tokenizer_vendor {

enum class PreTokenizerType {
    Llama3,
    Jais2,
    Dbrx,
    Smaug,
    DeepseekLlm,
    Deepseek3Llm,
    HunyuanDense,
    JoyaiLlm,
    Youtu,
    DeepseekCoder,
    Falcon,
    Starcoder,
    Refact,
    CommandR,
    Smollm,
    Codeshell,
    Exaone,
    Minerva,
    Gpt2,
    Mpt,
    Olmo,
    Jais,
    Trillion,
    GraniteDocling,
    StableLm2,
    Qwen2,
    Qwen35,
    Hunyuan,
    SolarOpen,
    Poro,
    Bloom,
    Gpt3Finnish,
    Chatglm4,
    Viking,
    Tekken,
    Chameleon,
    Gpt4o,
    MinimaxM2,
    TinyAya,
    KimiK2,
    SuperBpe,
    BailingMoe,
    SeedCoder,
    Grok2,
    Afmoe,
    ExaoneMoe,
    Gemma4,
    SarvamMoe,
};

enum TokenAttr : uint32_t {
    TOKEN_ATTR_UNKNOWN      = 1u << 0u,
    TOKEN_ATTR_CONTROL      = 1u << 1u,
    TOKEN_ATTR_USER_DEFINED = 1u << 2u,
    TOKEN_ATTR_LSTRIP       = 1u << 3u,
    TOKEN_ATTR_RSTRIP       = 1u << 4u,
};

struct TokenData {
    std::string text;
    uint32_t attr = 0;
};

struct BpeVocabulary {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<int32_t, TokenData> id_to_token;
    std::unordered_map<std::string, int32_t> bpe_ranks;
    std::vector<int32_t> special_tokens_sorted;
    std::optional<int32_t> configured_pad_token_id;
    PreTokenizerType pre_type = PreTokenizerType::Gpt2;

    int32_t text_to_token(const std::string & text) const;
    int find_bpe_rank(const std::string & left, const std::string & right) const;
    const TokenData & require_token_data(int32_t token_id) const;
};

void rebuild_special_tokens_cache(BpeVocabulary & vocab);
std::vector<int32_t> tokenize_bpe(const BpeVocabulary & vocab, const std::string & text, bool parse_special = true);
std::string decode_bpe(const BpeVocabulary & vocab, const std::vector<int32_t> & token_ids, bool skip_special_tokens);

}  // namespace llama_tokenizer_vendor

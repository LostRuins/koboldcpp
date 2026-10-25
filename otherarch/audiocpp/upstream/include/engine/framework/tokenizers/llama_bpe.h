#pragma once

#include "engine/framework/tokenizers/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::tokenizers {

enum class LlamaBpePreTokenizer {
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
    Hunyuan,
    SolarOpen,
    Qwen35,
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

struct LlamaBpeAddedToken {
    LlamaBpeAddedToken(
        std::string content_ = {},
        std::optional<int32_t> id_ = std::nullopt,
        bool special_ = true,
        bool user_defined_ = true,
        bool unknown_ = false,
        bool lstrip_ = false,
        bool rstrip_ = false)
        : content(std::move(content_)),
          id(id_),
          special(special_),
          user_defined(user_defined_),
          unknown(unknown_),
          lstrip(lstrip_),
          rstrip(rstrip_) {}

    std::string content;
    std::optional<int32_t> id;
    bool special = true;
    bool user_defined = true;
    bool unknown = false;
    bool lstrip = false;
    bool rstrip = false;
};

struct LlamaBpeTokenizerSpec {
    LlamaBpeTokenizerSpec(
        std::filesystem::path vocab_path_ = {},
        std::filesystem::path merges_path_ = {},
        std::filesystem::path tokenizer_config_path_ = {},
        std::optional<std::filesystem::path> tokenizer_json_path_ = std::nullopt,
        LlamaBpePreTokenizer pre_type_ = LlamaBpePreTokenizer::Gpt2,
        std::vector<LlamaBpeAddedToken> additional_special_tokens_ = {},
        std::string normalizer_space_replacement_ = {})
        : vocab_path(std::move(vocab_path_)),
          merges_path(std::move(merges_path_)),
          tokenizer_config_path(std::move(tokenizer_config_path_)),
          tokenizer_json_path(std::move(tokenizer_json_path_)),
          pre_type(pre_type_),
          additional_special_tokens(std::move(additional_special_tokens_)),
          normalizer_space_replacement(std::move(normalizer_space_replacement_)) {}

    std::filesystem::path vocab_path;
    std::filesystem::path merges_path;
    std::filesystem::path tokenizer_config_path;
    std::optional<std::filesystem::path> tokenizer_json_path;
    LlamaBpePreTokenizer pre_type = LlamaBpePreTokenizer::Gpt2;
    std::vector<LlamaBpeAddedToken> additional_special_tokens;
    std::string normalizer_space_replacement;
};

class LlamaBpeTokenizer final : public ITokenizer {
public:
    explicit LlamaBpeTokenizer(LlamaBpeTokenizerSpec spec);

    std::string family() const override;
    TokenizedText tokenize(const std::string & text) const override;

    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<int32_t> encode(const std::string & text, bool parse_special) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;

    std::optional<int32_t> find_token_id(const std::string & token) const;
    bool is_special_token_id(int32_t token_id) const;
    bool is_control_token_id(int32_t token_id) const;
    std::optional<int32_t> configured_pad_token_id() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

std::shared_ptr<LlamaBpeTokenizer> load_llama_bpe_tokenizer(const LlamaBpeTokenizerSpec & spec);

}  // namespace engine::tokenizers

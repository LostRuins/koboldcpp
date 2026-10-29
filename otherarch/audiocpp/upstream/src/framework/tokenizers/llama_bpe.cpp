#include "engine/framework/tokenizers/llama_bpe.h"

#include "engine/framework/io/json.h"

#include "bpe-core.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::tokenizers {
namespace {

namespace vendor = llama_tokenizer_vendor;

using vendor::BpeVocabulary;
using vendor::PreTokenizerType;
using vendor::TokenData;

PreTokenizerType convert_pre_type(const LlamaBpePreTokenizer pre_type) {
    switch (pre_type) {
        case LlamaBpePreTokenizer::Llama3: return PreTokenizerType::Llama3;
        case LlamaBpePreTokenizer::Jais2: return PreTokenizerType::Jais2;
        case LlamaBpePreTokenizer::Dbrx: return PreTokenizerType::Dbrx;
        case LlamaBpePreTokenizer::Smaug: return PreTokenizerType::Smaug;
        case LlamaBpePreTokenizer::DeepseekLlm: return PreTokenizerType::DeepseekLlm;
        case LlamaBpePreTokenizer::Deepseek3Llm: return PreTokenizerType::Deepseek3Llm;
        case LlamaBpePreTokenizer::HunyuanDense: return PreTokenizerType::HunyuanDense;
        case LlamaBpePreTokenizer::JoyaiLlm: return PreTokenizerType::JoyaiLlm;
        case LlamaBpePreTokenizer::Youtu: return PreTokenizerType::Youtu;
        case LlamaBpePreTokenizer::DeepseekCoder: return PreTokenizerType::DeepseekCoder;
        case LlamaBpePreTokenizer::Falcon: return PreTokenizerType::Falcon;
        case LlamaBpePreTokenizer::Starcoder: return PreTokenizerType::Starcoder;
        case LlamaBpePreTokenizer::Refact: return PreTokenizerType::Refact;
        case LlamaBpePreTokenizer::CommandR: return PreTokenizerType::CommandR;
        case LlamaBpePreTokenizer::Smollm: return PreTokenizerType::Smollm;
        case LlamaBpePreTokenizer::Codeshell: return PreTokenizerType::Codeshell;
        case LlamaBpePreTokenizer::Exaone: return PreTokenizerType::Exaone;
        case LlamaBpePreTokenizer::Minerva: return PreTokenizerType::Minerva;
        case LlamaBpePreTokenizer::Gpt2: return PreTokenizerType::Gpt2;
        case LlamaBpePreTokenizer::Mpt: return PreTokenizerType::Mpt;
        case LlamaBpePreTokenizer::Olmo: return PreTokenizerType::Olmo;
        case LlamaBpePreTokenizer::Jais: return PreTokenizerType::Jais;
        case LlamaBpePreTokenizer::Trillion: return PreTokenizerType::Trillion;
        case LlamaBpePreTokenizer::GraniteDocling: return PreTokenizerType::GraniteDocling;
        case LlamaBpePreTokenizer::StableLm2: return PreTokenizerType::StableLm2;
        case LlamaBpePreTokenizer::Qwen2: return PreTokenizerType::Qwen2;
        case LlamaBpePreTokenizer::Hunyuan: return PreTokenizerType::Hunyuan;
        case LlamaBpePreTokenizer::SolarOpen: return PreTokenizerType::SolarOpen;
        case LlamaBpePreTokenizer::Qwen35: return PreTokenizerType::Qwen35;
        case LlamaBpePreTokenizer::Poro: return PreTokenizerType::Poro;
        case LlamaBpePreTokenizer::Bloom: return PreTokenizerType::Bloom;
        case LlamaBpePreTokenizer::Gpt3Finnish: return PreTokenizerType::Gpt3Finnish;
        case LlamaBpePreTokenizer::Chatglm4: return PreTokenizerType::Chatglm4;
        case LlamaBpePreTokenizer::Viking: return PreTokenizerType::Viking;
        case LlamaBpePreTokenizer::Tekken: return PreTokenizerType::Tekken;
        case LlamaBpePreTokenizer::Chameleon: return PreTokenizerType::Chameleon;
        case LlamaBpePreTokenizer::Gpt4o: return PreTokenizerType::Gpt4o;
        case LlamaBpePreTokenizer::MinimaxM2: return PreTokenizerType::MinimaxM2;
        case LlamaBpePreTokenizer::TinyAya: return PreTokenizerType::TinyAya;
        case LlamaBpePreTokenizer::KimiK2: return PreTokenizerType::KimiK2;
        case LlamaBpePreTokenizer::SuperBpe: return PreTokenizerType::SuperBpe;
        case LlamaBpePreTokenizer::BailingMoe: return PreTokenizerType::BailingMoe;
        case LlamaBpePreTokenizer::SeedCoder: return PreTokenizerType::SeedCoder;
        case LlamaBpePreTokenizer::Grok2: return PreTokenizerType::Grok2;
        case LlamaBpePreTokenizer::Afmoe: return PreTokenizerType::Afmoe;
        case LlamaBpePreTokenizer::ExaoneMoe: return PreTokenizerType::ExaoneMoe;
        case LlamaBpePreTokenizer::Gemma4: return PreTokenizerType::Gemma4;
        case LlamaBpePreTokenizer::SarvamMoe: return PreTokenizerType::SarvamMoe;
    }
    throw std::runtime_error("unknown llama_bpe pre-tokenizer");
}

std::string pair_key(const std::string & left, const std::string & right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

uint32_t token_attr_from_json(const engine::io::json::Value & token_config) {
    // HF added tokens must be matched as whole tokens during encode even when
    // they are not marked "special". Model-local tokenizers historically did
    // this for Qwen-family assets, so keep that behavior in the shared bridge.
    uint32_t attr = vendor::TOKEN_ATTR_USER_DEFINED;
    const auto * special = token_config.find("special");
    if (special != nullptr && special->is_bool() && special->as_bool()) {
        attr |= vendor::TOKEN_ATTR_CONTROL;
    }
    const auto * lstrip = token_config.find("lstrip");
    if (lstrip != nullptr && lstrip->is_bool() && lstrip->as_bool()) {
        attr |= vendor::TOKEN_ATTR_LSTRIP;
    }
    const auto * rstrip = token_config.find("rstrip");
    if (rstrip != nullptr && rstrip->is_bool() && rstrip->as_bool()) {
        attr |= vendor::TOKEN_ATTR_RSTRIP;
    }
    return attr;
}

uint32_t token_attr_from_added_token(const LlamaBpeAddedToken & token) {
    uint32_t attr = 0;
    if (token.user_defined) {
        attr |= vendor::TOKEN_ATTR_USER_DEFINED;
    }
    if (token.special) {
        attr |= vendor::TOKEN_ATTR_CONTROL;
    }
    if (token.unknown) {
        attr |= vendor::TOKEN_ATTR_UNKNOWN;
    }
    if (token.lstrip) {
        attr |= vendor::TOKEN_ATTR_LSTRIP;
    }
    if (token.rstrip) {
        attr |= vendor::TOKEN_ATTR_RSTRIP;
    }
    return attr;
}

void load_vocab_json(const std::filesystem::path & vocab_path, BpeVocabulary & vocab) {
    const auto vocab_json = engine::io::json::parse_file(vocab_path);
    for (const auto & [token, id] : vocab_json.as_object()) {
        const int32_t token_id = static_cast<int32_t>(id.as_i64());
        vocab.token_to_id.emplace(token, token_id);
        vocab.id_to_token.emplace(token_id, TokenData{token, 0});
    }
}

void add_merge(BpeVocabulary & vocab, const std::string & merge, const int32_t rank) {
    const size_t split = merge.find(' ');
    if (split == std::string::npos) {
        throw std::runtime_error("invalid tokenizer merge entry: " + merge);
    }
    vocab.bpe_ranks.emplace(pair_key(merge.substr(0, split), merge.substr(split + 1)), rank);
}

void load_model_from_tokenizer_json(const engine::io::json::Value & tokenizer_json, BpeVocabulary & vocab) {
    const auto & model = tokenizer_json.require("model");
    if (model.require("type").as_string() != "BPE") {
        throw std::runtime_error("llama BPE tokenizer expects a BPE tokenizer.json model");
    }
    for (const auto & [token, id] : model.require("vocab").as_object()) {
        const int32_t token_id = static_cast<int32_t>(id.as_i64());
        vocab.token_to_id.emplace(token, token_id);
        vocab.id_to_token.emplace(token_id, TokenData{token, 0});
    }

    int32_t rank = 0;
    for (const auto & merge : model.require("merges").as_array()) {
        if (merge.is_string()) {
            add_merge(vocab, merge.as_string(), rank++);
            continue;
        }
        const auto & pair = merge.as_array();
        if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_string()) {
            throw std::runtime_error("tokenizer.json BPE merge must be a string or two-string array");
        }
        vocab.bpe_ranks.emplace(pair_key(pair[0].as_string(), pair[1].as_string()), rank++);
    }
}

void load_added_tokens_from_tokenizer_json(const engine::io::json::Value & tokenizer_json, BpeVocabulary & vocab) {
    const auto * added = tokenizer_json.find("added_tokens");
    if (added == nullptr || !added->is_array()) {
        return;
    }
    for (const auto & token_config : added->as_array()) {
        const auto * id = token_config.find("id");
        const auto * content = token_config.find("content");
        if (id == nullptr || !id->is_number() || content == nullptr || !content->is_string()) {
            continue;
        }
        const int32_t token_id = static_cast<int32_t>(id->as_i64());
        const std::string token_text = content->as_string();
        vocab.token_to_id[token_text] = token_id;
        vocab.id_to_token[token_id] = TokenData{token_text, token_attr_from_json(token_config)};
    }
}

void load_added_tokens_from_config(const std::filesystem::path & config_path, BpeVocabulary & vocab) {
    const auto tokenizer_config = engine::io::json::parse_file(config_path);
    if (const auto * added = tokenizer_config.find("added_tokens_decoder"); added != nullptr) {
        for (const auto & [id_text, token_config] : added->as_object()) {
            const auto * content = token_config.find("content");
            if (content == nullptr || !content->is_string()) {
                continue;
            }
            const int32_t token_id = static_cast<int32_t>(std::stoll(id_text));
            const std::string token_text = content->as_string();
            vocab.token_to_id[token_text] = token_id;
            vocab.id_to_token[token_id] = TokenData{token_text, token_attr_from_json(token_config)};
        }
    }

    const auto * pad_token = tokenizer_config.find("pad_token");
    if (pad_token != nullptr && pad_token->is_string()) {
        const auto it = vocab.token_to_id.find(pad_token->as_string());
        if (it != vocab.token_to_id.end()) {
            vocab.configured_pad_token_id = it->second;
        }
    }
}

void add_runtime_special_tokens(const std::vector<LlamaBpeAddedToken> & tokens, BpeVocabulary & vocab) {
    if (tokens.empty()) {
        return;
    }
    int32_t next_id = 0;
    for (const auto & [id, _] : vocab.id_to_token) {
        next_id = std::max(next_id, id + 1);
    }
    for (const auto & token : tokens) {
        if (token.content.empty()) {
            throw std::runtime_error("llama BPE runtime added token cannot be empty");
        }
        const auto existing_token = vocab.token_to_id.find(token.content);
        if (existing_token != vocab.token_to_id.end()) {
            if (token.id.has_value() && existing_token->second != *token.id) {
                throw std::runtime_error("llama BPE runtime added token id mismatch for token: " + token.content);
            }
            continue;
        }
        const int32_t token_id = token.id.value_or(next_id);
        const auto existing_id = vocab.id_to_token.find(token_id);
        if (existing_id != vocab.id_to_token.end()) {
            throw std::runtime_error("llama BPE runtime added token id collision for token: " + token.content);
        }
        vocab.token_to_id[token.content] = token_id;
        vocab.id_to_token[token_id] = TokenData{token.content, token_attr_from_added_token(token)};
        next_id = std::max(next_id, token_id + 1);
    }
}

void load_merges(const std::filesystem::path & merges_path, BpeVocabulary & vocab) {
    std::ifstream merges(merges_path);
    if (!merges) {
        throw std::runtime_error("failed to open tokenizer merges: " + merges_path.string());
    }
    std::string line;
    int32_t rank = 0;
    while (std::getline(merges, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        add_merge(vocab, line, rank++);
    }
}

std::string replace_spaces(std::string text, const std::string & replacement) {
    if (replacement.empty()) {
        return text;
    }
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        if (ch == ' ') {
            out += replacement;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

}  // namespace

struct LlamaBpeTokenizer::Impl {
    explicit Impl(const LlamaBpeTokenizerSpec & spec) {
        normalizer_space_replacement = spec.normalizer_space_replacement;
        vocab.pre_type = convert_pre_type(spec.pre_type);
        const bool has_vocab = !spec.vocab_path.empty();
        const bool has_merges = !spec.merges_path.empty();
        if (has_vocab != has_merges) {
            throw std::runtime_error("llama BPE tokenizer requires both vocab and merges paths");
        }
        if (has_vocab) {
            load_vocab_json(spec.vocab_path, vocab);
            load_merges(spec.merges_path, vocab);
        } else if (spec.tokenizer_json_path.has_value()) {
            const auto tokenizer_json = engine::io::json::parse_file(*spec.tokenizer_json_path);
            load_model_from_tokenizer_json(tokenizer_json, vocab);
            load_added_tokens_from_tokenizer_json(tokenizer_json, vocab);
        } else {
            throw std::runtime_error("llama BPE tokenizer requires vocab/merges or tokenizer.json");
        }
        if (has_vocab && spec.tokenizer_json_path.has_value()) {
            const auto tokenizer_json = engine::io::json::parse_file(*spec.tokenizer_json_path);
            load_added_tokens_from_tokenizer_json(tokenizer_json, vocab);
        }
        if (!spec.tokenizer_config_path.empty()) {
            load_added_tokens_from_config(spec.tokenizer_config_path, vocab);
        }
        add_runtime_special_tokens(spec.additional_special_tokens, vocab);
        vendor::rebuild_special_tokens_cache(vocab);
    }

    BpeVocabulary vocab;
    std::string normalizer_space_replacement;
};

LlamaBpeTokenizer::LlamaBpeTokenizer(LlamaBpeTokenizerSpec spec)
    : impl_(std::make_shared<Impl>(spec)) {}

std::string LlamaBpeTokenizer::family() const {
    return "llama_bpe";
}

TokenizedText LlamaBpeTokenizer::tokenize(const std::string & text) const {
    return TokenizedText{text, encode(text)};
}

std::vector<int32_t> LlamaBpeTokenizer::encode(const std::string & text) const {
    return vendor::tokenize_bpe(impl_->vocab, replace_spaces(text, impl_->normalizer_space_replacement), true);
}

std::vector<int32_t> LlamaBpeTokenizer::encode(const std::string & text, const bool parse_special) const {
    return vendor::tokenize_bpe(impl_->vocab, replace_spaces(text, impl_->normalizer_space_replacement), parse_special);
}

std::string LlamaBpeTokenizer::decode(const std::vector<int32_t> & token_ids, const bool skip_special_tokens) const {
    return vendor::decode_bpe(impl_->vocab, token_ids, skip_special_tokens);
}

std::optional<int32_t> LlamaBpeTokenizer::find_token_id(const std::string & token) const {
    const auto it = impl_->vocab.token_to_id.find(token);
    if (it == impl_->vocab.token_to_id.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool LlamaBpeTokenizer::is_special_token_id(const int32_t token_id) const {
    const auto it = impl_->vocab.id_to_token.find(token_id);
    if (it == impl_->vocab.id_to_token.end()) {
        return false;
    }
    return (it->second.attr & (vendor::TOKEN_ATTR_CONTROL | vendor::TOKEN_ATTR_USER_DEFINED | vendor::TOKEN_ATTR_UNKNOWN)) != 0;
}

bool LlamaBpeTokenizer::is_control_token_id(const int32_t token_id) const {
    const auto it = impl_->vocab.id_to_token.find(token_id);
    return it != impl_->vocab.id_to_token.end() && (it->second.attr & vendor::TOKEN_ATTR_CONTROL) != 0;
}

std::optional<int32_t> LlamaBpeTokenizer::configured_pad_token_id() const noexcept {
    return impl_->vocab.configured_pad_token_id;
}

std::shared_ptr<LlamaBpeTokenizer> load_llama_bpe_tokenizer(const LlamaBpeTokenizerSpec & spec) {
    return std::make_shared<LlamaBpeTokenizer>(spec);
}

}  // namespace engine::tokenizers

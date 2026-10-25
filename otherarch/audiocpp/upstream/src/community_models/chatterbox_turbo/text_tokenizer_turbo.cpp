#include "engine/community_models/chatterbox_turbo/text_tokenizer_turbo.h"

#include "engine/framework/io/json.h"

#include <cctype>
#include <unordered_set>

namespace engine::community_models::chatterbox_turbo {

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_chatterbox_turbo_tokenizer(
    const std::filesystem::path & vocab_path,
    const std::filesystem::path & merges_path,
    const std::filesystem::path & special_tokens_path) {
    std::vector<engine::tokenizers::LlamaBpeAddedToken> special_tokens;
    const auto special_tokens_json = engine::io::json::parse_file(special_tokens_path);
    for (const auto & entry : special_tokens_json.as_array()) {
        special_tokens.emplace_back(
            engine::io::json::require_string(entry, "token"),
            static_cast<int32_t>(engine::io::json::require_i64(entry, "id")));
    }

    engine::tokenizers::LlamaBpeTokenizerSpec spec(
        vocab_path,
        merges_path,
        /*tokenizer_config_path=*/{},
        /*tokenizer_json_path=*/std::nullopt,
        engine::tokenizers::LlamaBpePreTokenizer::Gpt2,
        std::move(special_tokens));
    return engine::tokenizers::load_llama_bpe_tokenizer(spec);
}

std::string chatterbox_turbo_punc_norm(const std::string & text) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }
    std::string normalized = text;
    if (std::islower(static_cast<unsigned char>(normalized.front()))) {
        normalized.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized.front())));
    }

    // Collapse runs of whitespace to single spaces (Python's " ".join(text.split())).
    {
        std::string collapsed;
        collapsed.reserve(normalized.size());
        bool in_space = false;
        for (char c : normalized) {
            const bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
            if (is_space) {
                in_space = true;
                continue;
            }
            if (in_space && !collapsed.empty()) {
                collapsed += ' ';
            }
            in_space = false;
            collapsed += c;
        }
        normalized = std::move(collapsed);
    }

    static const std::vector<std::pair<std::string, std::string>> kReplacements = {
        {"\xE2\x80\xA6", ", "},  // U+2026 HORIZONTAL ELLIPSIS
        {":", ","},
        {"\xE2\x80\x94", "-"},  // U+2014 EM DASH
        {"\xE2\x80\x93", "-"},  // U+2013 EN DASH
        {" ,", ","},
        {"\xE2\x80\x9C", "\""},  // U+201C LEFT DOUBLE QUOTATION MARK
        {"\xE2\x80\x9D", "\""},  // U+201D RIGHT DOUBLE QUOTATION MARK
        {"\xE2\x80\x98", "'"},   // U+2018 LEFT SINGLE QUOTATION MARK
        {"\xE2\x80\x99", "'"},   // U+2019 RIGHT SINGLE QUOTATION MARK
    };
    for (const auto & [from, to] : kReplacements) {
        size_t pos = 0;
        while ((pos = normalized.find(from, pos)) != std::string::npos) {
            normalized.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }
    static const std::unordered_set<char> kSentenceEnders = {'.', '!', '?', '-', ','};
    if (normalized.empty() || kSentenceEnders.find(normalized.back()) == kSentenceEnders.end()) {
        normalized += '.';
    }
    return normalized;
}

std::vector<int32_t> encode_chatterbox_turbo_text(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & text) {
    return tokenizer.encode(chatterbox_turbo_punc_norm(text), /*parse_special=*/true);
}

}  // namespace engine::community_models::chatterbox_turbo

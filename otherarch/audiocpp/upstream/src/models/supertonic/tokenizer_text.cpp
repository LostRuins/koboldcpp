#include "engine/models/supertonic/tokenizer_text.h"

#include "engine/framework/text/unicode_normalization.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace engine::models::supertonic {
namespace {

const std::unordered_set<std::string> kLanguages = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
};

std::shared_ptr<const SupertonicAssets> require_assets(std::shared_ptr<const SupertonicAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Supertonic tokenizer requires assets");
    }
    return assets;
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string trim_ascii_space(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ends_with_reference_punctuation(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    switch (last) {
        case '.':
        case '!':
        case '?':
        case ';':
        case ':':
        case ',':
        case '\'':
        case '"':
        case ')':
        case ']':
        case '}':
            return true;
        default:
            break;
    }
    static constexpr const char * suffixes[] = {"…", "。", "」", "』", "】", "〉", "》", "›", "»"};
    for (const char * suffix : suffixes) {
        const std::string marker(suffix);
        if (text.size() >= marker.size() && text.compare(text.size() - marker.size(), marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

SupertonicTextTokenizer::SupertonicTextTokenizer(std::shared_ptr<const SupertonicAssets> assets)
    : assets_(require_assets(std::move(assets))) {}

SupertonicTextInputs SupertonicTextTokenizer::encode(const std::string & text, const std::string & language) const {
    const auto processed = preprocess(text, language);
    const auto codepoints = engine::text::normalize_nfkd_codepoints(utf8_to_codepoints(processed));
    SupertonicTextInputs out;
    out.length = static_cast<int64_t>(codepoints.size());
    out.ids.reserve(codepoints.size());
    out.mask.assign(codepoints.size(), 1.0F);
    for (const uint32_t codepoint : codepoints) {
        const auto it = assets_->unicode_indexer.find(codepoint);
        if (it == assets_->unicode_indexer.end()) {
            throw std::runtime_error("Supertonic unicode indexer has no entry for codepoint " + std::to_string(codepoint));
        }
        out.ids.push_back(it->second);
    }
    return out;
}

std::string SupertonicTextTokenizer::preprocess(const std::string & text, const std::string & language) const {
    if (kLanguages.find(language) == kLanguages.end()) {
        throw std::runtime_error("invalid Supertonic language: " + language);
    }
    std::string result = text;
    const std::pair<const char *, const char *> replacements[] = {
        {"–", "-"}, {"‑", "-"}, {"—", "-"}, {"_", " "}, {"“", "\""}, {"”", "\""}, {"‘", "'"}, {"’", "'"},
        {"´", "'"}, {"`", "'"}, {"[", " "}, {"]", " "}, {"|", " "}, {"/", " "}, {"#", " "}, {"→", " "},
        {"←", " "}, {"@", " at "}, {"e.g.,", "for example, "}, {"i.e.,", "that is, "},
    };
    for (const auto & replacement : replacements) {
        replace_all(result, replacement.first, replacement.second);
    }
    const char * remove_symbols[] = {"♥", "☆", "♡", "©", "\\"};
    for (const char * symbol : remove_symbols) {
        replace_all(result, symbol, "");
    }
    result = std::regex_replace(result, std::regex(" ,"), ",");
    result = std::regex_replace(result, std::regex(" \\."), ".");
    result = std::regex_replace(result, std::regex(" !"), "!");
    result = std::regex_replace(result, std::regex(" \\?"), "?");
    result = std::regex_replace(result, std::regex(" ;"), ";");
    result = std::regex_replace(result, std::regex(" :"), ":");
    result = std::regex_replace(result, std::regex(" '"), "'");
    while (result.find("\"\"") != std::string::npos) {
        replace_all(result, "\"\"", "\"");
    }
    while (result.find("''") != std::string::npos) {
        replace_all(result, "''", "'");
    }
    result = std::regex_replace(result, std::regex("\\s+"), " ");
    result = trim_ascii_space(result);
    if (!ends_with_reference_punctuation(result)) {
        result += ".";
    }
    return "<" + language + ">" + result + "</" + language + ">";
}

std::vector<uint32_t> SupertonicTextTokenizer::utf8_to_codepoints(const std::string & text) const {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        size_t width = 0;
        if ((c & 0x80U) == 0U) {
            codepoint = c;
            width = 1;
        } else if ((c & 0xE0U) == 0xC0U && i + 1 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x1FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU);
            width = 2;
        } else if ((c & 0xF0U) == 0xE0U && i + 2 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x0FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU);
            width = 3;
        } else if ((c & 0xF8U) == 0xF0U && i + 3 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x07U) << 18U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3FU);
            width = 4;
        } else {
            throw std::runtime_error("Supertonic input contains invalid UTF-8");
        }
        out.push_back(codepoint);
        i += width;
    }
    return out;
}

}  // namespace engine::models::supertonic

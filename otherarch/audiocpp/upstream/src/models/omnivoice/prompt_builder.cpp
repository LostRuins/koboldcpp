#include "engine/models/omnivoice/prompt_builder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

constexpr char kReferenceDurationFallbackText[] = "Nice to meet you.";
constexpr int64_t kReferenceDurationFallbackFrames = 25;
constexpr char kSplitPunctuationAscii[] = ".,;:!?";
constexpr char kClosingMarksAscii[] = "\"')]>";

const std::unordered_set<std::string> & split_punctuation_utf8() {
    static const std::unordered_set<std::string> marks = {"。", "，", "；", "：", "！", "？"};
    return marks;
}

const std::unordered_set<std::string> & closing_marks_utf8() {
    static const std::unordered_set<std::string> marks = {"）", "》", "」", "】", "”", "’"};
    return marks;
}

const std::unordered_set<std::string> & abbreviations() {
    static const std::unordered_set<std::string> items = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
        "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.",
        "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.",
        "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.",
        "Etc.", "approx.", "fig.", "def."
    };
    return items;
}

bool contains_chinese(std::string_view text) {
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        size_t len = 1;
        if ((ch & 0x80U) == 0) {
            codepoint = ch;
        } else if ((ch & 0xE0U) == 0xC0U && i + 1 < text.size()) {
            len = 2;
            codepoint = ((static_cast<uint32_t>(ch) & 0x1FU) << 6) |
                (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU);
        } else if ((ch & 0xF0U) == 0xE0U && i + 2 < text.size()) {
            len = 3;
            codepoint = ((static_cast<uint32_t>(ch) & 0x0FU) << 12) |
                ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU) << 6) |
                (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2])) & 0x3FU);
        } else if ((ch & 0xF8U) == 0xF0U && i + 3 < text.size()) {
            len = 4;
            codepoint = ((static_cast<uint32_t>(ch) & 0x07U) << 18) |
                ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU) << 12) |
                ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2])) & 0x3FU) << 6) |
                (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3])) & 0x3FU);
        } else {
            throw std::runtime_error("OmniVoice prompt builder encountered invalid UTF-8");
        }
        if (codepoint >= 0x4E00U && codepoint <= 0x9FFFU) {
            return true;
        }
        i += len;
    }
    return false;
}

bool decode_codepoint(std::string_view text, size_t offset, uint32_t & codepoint, size_t & len) {
    if (offset >= text.size()) {
        return false;
    }
    const unsigned char ch = static_cast<unsigned char>(text[offset]);
    len = 1;
    if ((ch & 0x80U) == 0) {
        codepoint = ch;
    } else if ((ch & 0xE0U) == 0xC0U && offset + 1 < text.size()) {
        len = 2;
        codepoint = ((static_cast<uint32_t>(ch) & 0x1FU) << 6) |
            (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 1])) & 0x3FU);
    } else if ((ch & 0xF0U) == 0xE0U && offset + 2 < text.size()) {
        len = 3;
        codepoint = ((static_cast<uint32_t>(ch) & 0x0FU) << 12) |
            ((static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 1])) & 0x3FU) << 6) |
            (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 2])) & 0x3FU);
    } else if ((ch & 0xF8U) == 0xF0U && offset + 3 < text.size()) {
        len = 4;
        codepoint = ((static_cast<uint32_t>(ch) & 0x07U) << 18) |
            ((static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 1])) & 0x3FU) << 12) |
            ((static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 2])) & 0x3FU) << 6) |
            (static_cast<uint32_t>(static_cast<unsigned char>(text[offset + 3])) & 0x3FU);
    } else {
        throw std::runtime_error("OmniVoice prompt builder encountered invalid UTF-8");
    }
    return true;
}

std::string lowercase_ascii(std::string_view text) {
    std::string out(text);
    for (char & ch : out) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch + ('a' - 'A'));
        }
    }
    return out;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
        text.substr(text.size() - suffix.size(), suffix.size()) == suffix;
}

std::string trim_copy(std::string_view text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
        ++start;
    }
    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(start, end - start));
}

std::string add_punctuation(const std::string & text) {
    static const std::unordered_set<std::string> punctuation = {
        ";", ":", ",", ".", "!", "?", "...", ")", "]", "}", "\"", "'",
        "；", "：", "，", "。", "！", "？", "、", "……", "）", "】"};
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return trimmed;
    }
    const std::string tail = trimmed.size() >= 3 ? trimmed.substr(trimmed.size() - 3) : trimmed;
    if (punctuation.count(std::string(1, trimmed.back())) != 0 || punctuation.count(tail) != 0) {
        return trimmed;
    }
    return trimmed + (contains_chinese(trimmed) ? "。" : ".");
}

bool is_split_punctuation(std::string_view token) {
    if (token.size() == 1 && std::strchr(kSplitPunctuationAscii, token.front()) != nullptr) {
        return true;
    }
    return split_punctuation_utf8().count(std::string(token)) != 0;
}

bool is_closing_mark(std::string_view token) {
    if (token.size() == 1 && std::strchr(kClosingMarksAscii, token.front()) != nullptr) {
        return true;
    }
    return closing_marks_utf8().count(std::string(token)) != 0;
}

bool is_abbreviation_sentence(const std::vector<std::string> & sentence) {
    std::string text;
    for (const auto & token : sentence) {
        text += token;
    }
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return false;
    }
    const auto last_space = trimmed.find_last_of(' ');
    const std::string last_word = last_space == std::string::npos ? trimmed : trimmed.substr(last_space + 1);
    return abbreviations().count(last_word) != 0;
}

std::string combine_text(const std::string & text, const std::optional<std::string> & ref_text) {
    std::string full_text;
    if (ref_text.has_value() && !trim_copy(*ref_text).empty()) {
        full_text = trim_copy(*ref_text) + " " + trim_copy(text);
    } else {
        full_text = trim_copy(text);
    }

    full_text = std::regex_replace(full_text, std::regex(R"([\r\n]+)"), "");
    size_t pos = 0;
    while ((pos = full_text.find("\xEF\xBC\x88", pos)) != std::string::npos) {
        full_text.replace(pos, 3, "(");
    }
    pos = 0;
    while ((pos = full_text.find("\xEF\xBC\x89", pos)) != std::string::npos) {
        full_text.replace(pos, 3, ")");
    }
    full_text = std::regex_replace(full_text, std::regex(R"([ \t]+)"), " ");
    std::vector<std::pair<uint32_t, std::string>> codepoints;
    for (size_t i = 0; i < full_text.size();) {
        uint32_t codepoint = 0;
        size_t len = 0;
        decode_codepoint(full_text, i, codepoint, len);
        codepoints.emplace_back(codepoint, full_text.substr(i, len));
        i += len;
    }

    std::string compact;
    compact.reserve(full_text.size());
    for (size_t i = 0; i < codepoints.size(); ++i) {
        const auto & [codepoint, bytes] = codepoints[i];
        if (codepoint == 32U) {
            const bool chinese_left = i > 0 && codepoints[i - 1].first >= 0x4E00U && codepoints[i - 1].first <= 0x9FFFU;
            const bool chinese_right =
                i + 1 < codepoints.size() && codepoints[i + 1].first >= 0x4E00U && codepoints[i + 1].first <= 0x9FFFU;
            if (!chinese_left && !chinese_right && (compact.empty() || compact.back() != ' ')) {
                compact.push_back(' ');
            }
            continue;
        }
        compact += bytes;
    }
    return compact;
}

struct LanguageMap {
    std::unordered_map<std::string, std::string> name_to_id;
    std::unordered_set<std::string> ids;
};

struct LanguageMapEntry {
    std::string_view id;
    std::string_view name;
};

constexpr LanguageMapEntry kLanguageMapEntries[] = {
#include "language_map.inc"
};

const LanguageMap & load_language_map() {
    static const LanguageMap map = [] {
        LanguageMap out;
        for (const auto & entry : kLanguageMapEntries) {
            const std::string id = trim_copy(entry.id);
            const std::string name = lowercase_ascii(trim_copy(entry.name));
            if (id.empty() || name.empty()) {
                continue;
            }
            out.ids.insert(id);
            out.name_to_id.emplace(name, id);
        }
        return out;
    }();
    return map;
}

std::optional<std::string> resolve_language(const OmniVoiceAssets & assets, const std::string & language) {
    (void) assets;
    const std::string trimmed = trim_copy(language);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    const std::string lowered = lowercase_ascii(trimmed);
    if (lowered == "none" || lowered == "auto") {
        return std::nullopt;
    }
    const auto & map = load_language_map();
    if (map.ids.find(trimmed) != map.ids.end()) {
        return trimmed;
    }
    const auto name_it = map.name_to_id.find(lowered);
    if (name_it != map.name_to_id.end()) {
        return name_it->second;
    }
    throw std::runtime_error(
        "unsupported OmniVoice language '" + language +
        "'; use an ISO language id like 'en' or a built-in OmniVoice language name");
}

int levenshtein_distance(std::string_view a, std::string_view b) {
    std::vector<int> prev(b.size() + 1);
    std::vector<int> curr(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = static_cast<int>(j);
    }
    for (size_t i = 1; i <= a.size(); ++i) {
        curr[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev.back();
}

std::optional<std::string> closest_match(std::string_view item, const std::unordered_set<std::string> & candidates) {
    std::optional<std::string> best;
    int best_distance = std::numeric_limits<int>::max();
    for (const auto & candidate : candidates) {
        const int distance = levenshtein_distance(item, candidate);
        if (distance < best_distance) {
            best_distance = distance;
            best = candidate;
        }
    }
    if (best.has_value() && best_distance <= std::max<int>(2, static_cast<int>(item.size() / 3))) {
        return best;
    }
    return std::nullopt;
}

const std::unordered_map<std::string, std::string> & instruct_en_to_zh() {
    static const std::unordered_map<std::string, std::string> map = {
        {"male", "男"},
        {"female", "女"},
        {"child", "儿童"},
        {"teenager", "少年"},
        {"young adult", "青年"},
        {"middle-aged", "中年"},
        {"elderly", "老年"},
        {"very low pitch", "极低音调"},
        {"low pitch", "低音调"},
        {"moderate pitch", "中音调"},
        {"high pitch", "高音调"},
        {"very high pitch", "极高音调"},
        {"whisper", "耳语"},
    };
    return map;
}

const std::unordered_map<std::string, std::string> & instruct_zh_to_en() {
    static std::unordered_map<std::string, std::string> map = [] {
        std::unordered_map<std::string, std::string> out;
        for (const auto & [en, zh] : instruct_en_to_zh()) {
            out.emplace(zh, en);
        }
        return out;
    }();
    return map;
}

const std::vector<std::unordered_set<std::string>> & mutually_exclusive_categories() {
    static const std::vector<std::unordered_set<std::string>> categories = {
        {"male", "female", "男", "女"},
        {"child", "teenager", "young adult", "middle-aged", "elderly", "儿童", "少年", "青年", "中年", "老年"},
        {"very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
         "极低音调", "低音调", "中音调", "高音调", "极高音调"},
        {"whisper", "耳语"},
        {"american accent", "british accent", "australian accent", "chinese accent", "canadian accent",
         "indian accent", "korean accent", "portuguese accent", "russian accent", "japanese accent"},
        {"河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"},
    };
    return categories;
}

const std::unordered_set<std::string> & all_valid_instructs() {
    static const std::unordered_set<std::string> items = [] {
        std::unordered_set<std::string> out;
        for (const auto & [en, zh] : instruct_en_to_zh()) {
            out.insert(en);
            out.insert(zh);
        }
        for (const auto & item : mutually_exclusive_categories()[4]) {
            out.insert(item);
        }
        for (const auto & item : mutually_exclusive_categories()[5]) {
            out.insert(item);
        }
        return out;
    }();
    return items;
}

std::string resolve_instruct(const std::string & instruct, bool use_zh) {
    const std::string trimmed = trim_copy(instruct);
    if (trimmed.empty()) {
        return {};
    }

    std::vector<std::string> raw_items;
    std::regex separator(R"(\s*[,，]\s*)");
    std::sregex_token_iterator begin(trimmed.begin(), trimmed.end(), separator, -1);
    std::sregex_token_iterator end;
    for (auto it = begin; it != end; ++it) {
        const std::string item = trim_copy(it->str());
        if (!item.empty()) {
            raw_items.push_back(lowercase_ascii(item));
        }
    }

    const auto & all_valid = all_valid_instructs();
    std::vector<std::string> normalised;
    std::vector<std::string> unknown_lines;
    for (const auto & item : raw_items) {
        if (all_valid.count(item) != 0) {
            normalised.push_back(item);
            continue;
        }
        const auto suggestion = closest_match(item, all_valid);
        if (suggestion.has_value()) {
            unknown_lines.push_back("'" + item + "' (unsupported; did you mean '" + *suggestion + "'?)");
        } else {
            unknown_lines.push_back("'" + item + "' (unsupported)");
        }
    }
    if (!unknown_lines.empty()) {
        std::string message = "unsupported OmniVoice instruct items: ";
        for (size_t i = 0; i < unknown_lines.size(); ++i) {
            if (i != 0) {
                message += ", ";
            }
            message += unknown_lines[i];
        }
        throw std::runtime_error(message);
    }

    const bool has_dialect = std::any_of(normalised.begin(), normalised.end(), [](const std::string & item) {
        return ends_with(item, "话");
    });
    const bool has_accent = std::any_of(normalised.begin(), normalised.end(), [](const std::string & item) {
        return item.find(" accent") != std::string::npos;
    });
    if (has_dialect && has_accent) {
        throw std::runtime_error("OmniVoice instruct cannot mix Chinese dialect and English accent");
    }
    if (has_dialect) {
        use_zh = true;
    } else if (has_accent) {
        use_zh = false;
    }

    std::vector<std::string> unified;
    unified.reserve(normalised.size());
    for (const auto & item : normalised) {
        if (use_zh) {
            const auto it = instruct_en_to_zh().find(item);
            unified.push_back(it != instruct_en_to_zh().end() ? it->second : item);
        } else {
            const auto it = instruct_zh_to_en().find(item);
            unified.push_back(it != instruct_zh_to_en().end() ? it->second : item);
        }
    }

    for (const auto & category : mutually_exclusive_categories()) {
        std::vector<std::string> hits;
        for (const auto & item : unified) {
            if (category.count(item) != 0) {
                hits.push_back(item);
            }
        }
        if (hits.size() > 1) {
            std::string message = "conflicting OmniVoice instruct items in the same category: ";
            for (size_t i = 0; i < hits.size(); ++i) {
                if (i != 0) {
                    message += " vs ";
                }
                message += "'" + hits[i] + "'";
            }
            throw std::runtime_error(message);
        }
    }

    const bool output_zh = std::any_of(unified.begin(), unified.end(), [](const std::string & item) {
        return contains_chinese(item);
    });
    const std::string separator_text = output_zh ? "，" : ", ";
    std::string out;
    for (size_t i = 0; i < unified.size(); ++i) {
        if (i != 0) {
            out += separator_text;
        }
        out += unified[i];
    }
    return out;
}

class RuleDurationEstimator {
public:
    RuleDurationEstimator() {
        weights_ = {
            {"cjk", 3.0}, {"hangul", 2.5}, {"kana", 2.2}, {"ethiopic", 3.0}, {"yi", 3.0},
            {"indic", 1.8}, {"thai_lao", 1.5}, {"khmer_myanmar", 1.8}, {"arabic", 1.5},
            {"hebrew", 1.5}, {"latin", 1.0}, {"cyrillic", 1.0}, {"greek", 1.0}, {"armenian", 1.0},
            {"georgian", 1.0}, {"punctuation", 0.5}, {"space", 0.2}, {"digit", 3.5}, {"mark", 0.0},
            {"default", 1.0},
        };
        ranges_ = {
            {0x02AF, "latin"}, {0x03FF, "greek"}, {0x052F, "cyrillic"}, {0x058F, "armenian"},
            {0x05FF, "hebrew"}, {0x077F, "arabic"}, {0x089F, "arabic"}, {0x08FF, "arabic"},
            {0x097F, "indic"}, {0x09FF, "indic"}, {0x0A7F, "indic"}, {0x0AFF, "indic"},
            {0x0B7F, "indic"}, {0x0BFF, "indic"}, {0x0C7F, "indic"}, {0x0CFF, "indic"},
            {0x0D7F, "indic"}, {0x0DFF, "indic"}, {0x0EFF, "thai_lao"}, {0x0FFF, "indic"},
            {0x109F, "khmer_myanmar"}, {0x10FF, "georgian"}, {0x11FF, "hangul"}, {0x137F, "ethiopic"},
            {0x139F, "ethiopic"}, {0x13FF, "default"}, {0x167F, "default"}, {0x169F, "default"},
            {0x16FF, "default"}, {0x171F, "default"}, {0x173F, "default"}, {0x175F, "default"},
            {0x177F, "default"}, {0x17FF, "khmer_myanmar"}, {0x18AF, "default"}, {0x18FF, "default"},
            {0x194F, "indic"}, {0x19DF, "indic"}, {0x19FF, "khmer_myanmar"}, {0x1A1F, "indic"},
            {0x1AAF, "indic"}, {0x1B7F, "indic"}, {0x1BBF, "indic"}, {0x1BFF, "indic"},
            {0x1C4F, "indic"}, {0x1C7F, "indic"}, {0x1C8F, "cyrillic"}, {0x1CBF, "georgian"},
            {0x1CCF, "indic"}, {0x1CFF, "indic"}, {0x1D7F, "latin"}, {0x1DBF, "latin"},
            {0x1DFF, "default"}, {0x1EFF, "latin"}, {0x309F, "kana"}, {0x30FF, "kana"},
            {0x312F, "cjk"}, {0x318F, "hangul"}, {0x9FFF, "cjk"}, {0xA4CF, "yi"},
            {0xA4FF, "default"}, {0xA63F, "default"}, {0xA69F, "cyrillic"}, {0xA6FF, "default"},
            {0xA7FF, "latin"}, {0xA82F, "indic"}, {0xA87F, "default"}, {0xA8DF, "indic"},
            {0xA8FF, "indic"}, {0xA92F, "indic"}, {0xA95F, "indic"}, {0xA97F, "hangul"},
            {0xA9DF, "indic"}, {0xA9FF, "khmer_myanmar"}, {0xAA5F, "indic"}, {0xAA7F, "khmer_myanmar"},
            {0xAADF, "indic"}, {0xAAFF, "indic"}, {0xAB2F, "ethiopic"}, {0xAB6F, "latin"},
            {0xABBF, "default"}, {0xABFF, "indic"}, {0xD7AF, "hangul"}, {0xFAFF, "cjk"},
            {0xFDFF, "arabic"}, {0xFE6F, "default"}, {0xFEFF, "arabic"}, {0xFFEF, "latin"},
        };
        breakpoints_.reserve(ranges_.size());
        for (const auto & [code, _] : ranges_) {
            breakpoints_.push_back(code);
        }
    }

    double estimate_duration(
        const std::string & target_text,
        const std::string & ref_text,
        double ref_duration,
        std::optional<double> low_threshold = 50.0,
        double boost_strength = 3.0) const {
        if (ref_duration <= 0.0 || ref_text.empty()) {
            return 0.0;
        }
        const double ref_weight = calculate_total_weight(ref_text);
        if (ref_weight == 0.0) {
            return 0.0;
        }
        const double speed_factor = ref_weight / ref_duration;
        const double target_weight = calculate_total_weight(target_text);
        double estimated_duration = target_weight / speed_factor;
        if (low_threshold.has_value() && estimated_duration < *low_threshold) {
            const double alpha = 1.0 / boost_strength;
            return *low_threshold * std::pow(estimated_duration / *low_threshold, alpha);
        }
        return estimated_duration;
    }

private:
    std::unordered_map<std::string, double> weights_;
    std::vector<std::pair<uint32_t, std::string>> ranges_;
    std::vector<uint32_t> breakpoints_;

    double char_weight(uint32_t codepoint) const {
        if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z')) {
            return weights_.at("latin");
        }
        if (codepoint == 32) {
            return weights_.at("space");
        }
        if (codepoint == 0x0640) {
            return weights_.at("mark");
        }
        if ((codepoint >= '0' && codepoint <= '9')) {
            return weights_.at("digit");
        }
        if (codepoint == 0x3000U ||
            codepoint == 0x00A0U ||
            (codepoint >= 0x0009U && codepoint <= 0x000DU) ||
            (codepoint >= 0x2000U && codepoint <= 0x200AU)) {
            return weights_.at("space");
        }
        if ((codepoint >= 0x0300 && codepoint <= 0x036F) ||
            (codepoint >= 0x1AB0 && codepoint <= 0x1AFF) ||
            (codepoint >= 0x1DC0 && codepoint <= 0x1DFF) ||
            (codepoint >= 0x20D0 && codepoint <= 0x20FF) ||
            (codepoint >= 0xFE20 && codepoint <= 0xFE2F)) {
            return weights_.at("mark");
        }
        if ((codepoint >= 33 && codepoint <= 47) || (codepoint >= 58 && codepoint <= 64) ||
            (codepoint >= 91 && codepoint <= 96) || (codepoint >= 123 && codepoint <= 126) ||
            (codepoint >= 0x2000 && codepoint <= 0x206F) ||
            (codepoint >= 0x3001 && codepoint <= 0x303F) ||
            (codepoint >= 0xFF01 && codepoint <= 0xFF0F) ||
            (codepoint >= 0xFF1A && codepoint <= 0xFF20) ||
            (codepoint >= 0xFF3B && codepoint <= 0xFF40) ||
            (codepoint >= 0xFF5B && codepoint <= 0xFF65)) {
            return weights_.at("punctuation");
        }

        const auto it = std::lower_bound(breakpoints_.begin(), breakpoints_.end(), codepoint);
        if (it != breakpoints_.end()) {
            const size_t index = static_cast<size_t>(it - breakpoints_.begin());
            const auto weight_it = weights_.find(ranges_[index].second);
            return weight_it != weights_.end() ? weight_it->second : weights_.at("default");
        }
        if (codepoint > 0x20000U) {
            return weights_.at("cjk");
        }
        return weights_.at("default");
    }

    double calculate_total_weight(const std::string & text) const {
        double total = 0.0;
        for (size_t i = 0; i < text.size();) {
            const unsigned char ch = static_cast<unsigned char>(text[i]);
            uint32_t codepoint = 0;
            size_t len = 1;
            if ((ch & 0x80U) == 0) {
                codepoint = ch;
            } else if ((ch & 0xE0U) == 0xC0U && i + 1 < text.size()) {
                len = 2;
                codepoint = ((static_cast<uint32_t>(ch) & 0x1FU) << 6) |
                    (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU);
            } else if ((ch & 0xF0U) == 0xE0U && i + 2 < text.size()) {
                len = 3;
                codepoint = ((static_cast<uint32_t>(ch) & 0x0FU) << 12) |
                    ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU) << 6) |
                    (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2])) & 0x3FU);
            } else if ((ch & 0xF8U) == 0xF0U && i + 3 < text.size()) {
                len = 4;
                codepoint = ((static_cast<uint32_t>(ch) & 0x07U) << 18) |
                    ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1])) & 0x3FU) << 12) |
                    ((static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2])) & 0x3FU) << 6) |
                    (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3])) & 0x3FU);
            } else {
                throw std::runtime_error("OmniVoice duration estimator encountered invalid UTF-8");
            }
            total += char_weight(codepoint);
            i += len;
        }
        return total;
    }
};

int64_t frame_rate(const OmniVoiceAssets & assets) {
    if (assets.config.audio_tokenizer.hop_length <= 0 || assets.config.audio_tokenizer.sample_rate <= 0) {
        throw std::runtime_error("OmniVoice audio tokenizer config has invalid sample_rate/hop_length");
    }
    return static_cast<int64_t>(std::llround(
        static_cast<double>(assets.config.audio_tokenizer.sample_rate) /
        static_cast<double>(assets.config.audio_tokenizer.hop_length)));
}

int64_t estimate_target_tokens(
    const OmniVoiceAssets & assets,
    const std::string & text,
    const std::optional<std::string> & ref_text,
    const std::optional<int64_t> & ref_audio_tokens,
    float speed) {
    static const RuleDurationEstimator estimator;
    const std::string effective_ref_text =
        ref_text.has_value() && !ref_text->empty() ? *ref_text : std::string(kReferenceDurationFallbackText);
    const int64_t effective_ref_frames =
        ref_audio_tokens.has_value() && *ref_audio_tokens > 0 ? *ref_audio_tokens : kReferenceDurationFallbackFrames;
    double estimate = estimator.estimate_duration(text, effective_ref_text, static_cast<double>(effective_ref_frames));
    if (speed > 0.0F && speed != 1.0F) {
        estimate /= static_cast<double>(speed);
    }
    (void) assets;
    return std::max<int64_t>(1, static_cast<int64_t>(estimate));
}

std::vector<std::string> chunk_text_punctuation_impl(
    std::string_view text,
    int64_t chunk_len,
    std::optional<int64_t> min_chunk_len) {
    if (chunk_len <= 0) {
        throw std::runtime_error("OmniVoice chunk_text_punctuation requires positive chunk_len");
    }

    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        uint32_t codepoint = 0;
        size_t len = 0;
        decode_codepoint(text, i, codepoint, len);
        tokens.emplace_back(text.substr(i, len));
        i += len;
    }

    std::vector<std::vector<std::string>> sentences;
    std::vector<std::string> current_sentence;
    for (const auto & token : tokens) {
        if (current_sentence.empty() && !sentences.empty() &&
            (is_split_punctuation(token) || is_closing_mark(token))) {
            sentences.back().push_back(token);
            continue;
        }
        current_sentence.push_back(token);
        if (is_split_punctuation(token)) {
            const bool abbreviation = token == "." && is_abbreviation_sentence(current_sentence);
            if (!abbreviation) {
                sentences.push_back(current_sentence);
                current_sentence.clear();
            }
        }
    }
    if (!current_sentence.empty()) {
        sentences.push_back(current_sentence);
    }

    std::vector<std::vector<std::string>> merged_chunks;
    std::vector<std::string> current_chunk;
    for (const auto & sentence : sentences) {
        if (static_cast<int64_t>(current_chunk.size() + sentence.size()) <= chunk_len) {
            current_chunk.insert(current_chunk.end(), sentence.begin(), sentence.end());
        } else {
            if (!current_chunk.empty()) {
                merged_chunks.push_back(current_chunk);
            }
            current_chunk = sentence;
        }
    }
    if (!current_chunk.empty()) {
        merged_chunks.push_back(current_chunk);
    }

    std::vector<std::vector<std::string>> final_chunks;
    if (min_chunk_len.has_value()) {
        const bool first_short = !merged_chunks.empty() &&
            static_cast<int64_t>(merged_chunks.front().size()) < *min_chunk_len;
        for (size_t i = 0; i < merged_chunks.size(); ++i) {
            auto & chunk = merged_chunks[i];
            if (i == 1 && first_short) {
                final_chunks.back().insert(final_chunks.back().end(), chunk.begin(), chunk.end());
                continue;
            }
            if (static_cast<int64_t>(chunk.size()) >= *min_chunk_len) {
                final_chunks.push_back(chunk);
            } else if (final_chunks.empty()) {
                final_chunks.push_back(chunk);
            } else {
                final_chunks.back().insert(final_chunks.back().end(), chunk.begin(), chunk.end());
            }
        }
    } else {
        final_chunks = std::move(merged_chunks);
    }

    std::vector<std::string> out;
    out.reserve(final_chunks.size());
    for (const auto & chunk : final_chunks) {
        std::string joined;
        for (const auto & token : chunk) {
            joined += token;
        }
        joined = trim_copy(joined);
        if (!joined.empty()) {
            out.push_back(std::move(joined));
        }
    }
    return out;
}

}  // namespace

OmniVoicePromptBuilder::OmniVoicePromptBuilder(
    std::shared_ptr<const OmniVoiceAssets> assets,
    const OmniVoiceTextTokenizer & tokenizer)
    : assets_(std::move(assets)),
      tokenizer_(tokenizer) {
    if (assets_ == nullptr) {
        throw std::runtime_error("OmniVoice prompt builder requires assets");
    }
}

OmniVoicePrompt OmniVoicePromptBuilder::build(const OmniVoiceRequest & request) const {
    if (trim_copy(request.text).empty()) {
        throw std::runtime_error("OmniVoice prompt builder requires non-empty text");
    }

    OmniVoicePrompt prompt;
    prompt.text = trim_copy(request.text);
    prompt.reference_rms = request.reference_rms;

    const auto resolved_language = resolve_language(*assets_, request.language);
    prompt.language = resolved_language.has_value() ? *resolved_language : std::string();

    if (request.reference_audio_tokens.has_value()) {
        prompt.mode = OmniVoicePromptMode::VoiceClone;
        if (trim_copy(request.reference_text).empty()) {
            throw std::runtime_error(
                "OmniVoice native voice clone currently requires reference_text when reference audio is provided");
        }
        prompt.reference_audio_tokens = request.reference_audio_tokens;
        prompt.reference_text = request.generation.preprocess_prompt
            ? add_punctuation(request.reference_text)
            : trim_copy(request.reference_text);
    } else if (!trim_copy(request.instruct).empty()) {
        prompt.mode = OmniVoicePromptMode::VoiceDesign;
    } else {
        prompt.mode = OmniVoicePromptMode::AutoVoice;
    }

    if (!trim_copy(request.instruct).empty()) {
        prompt.instruct = resolve_instruct(request.instruct, contains_chinese(prompt.text));
    }

    prompt.target_audio_tokens = engine::models::omnivoice::estimate_target_tokens(
        *assets_,
        prompt.text,
        prompt.reference_text.empty() ? std::optional<std::string>() : std::make_optional(prompt.reference_text),
        prompt.reference_audio_tokens.has_value()
            ? std::make_optional(prompt.reference_audio_tokens->frames)
            : std::optional<int64_t>(),
        request.generation.speed);

    if (request.generation.duration_seconds.has_value()) {
        prompt.target_audio_tokens = std::max<int64_t>(
            1,
                static_cast<int64_t>(std::llround(
                static_cast<double>(*request.generation.duration_seconds) *
                static_cast<double>(engine::models::omnivoice::frame_rate(*assets_)))));
    }

    std::string style_text;
    if (request.generation.denoise && prompt.reference_audio_tokens.has_value()) {
        style_text += "<|denoise|>";
    }
    style_text += "<|lang_start|>";
    style_text += prompt.language.empty() ? "None" : prompt.language;
    style_text += "<|lang_end|>";
    style_text += "<|instruct_start|>";
    style_text += prompt.instruct.empty() ? "None" : prompt.instruct;
    style_text += "<|instruct_end|>";

    const std::string combined = combine_text(
        prompt.text,
        prompt.reference_text.empty() ? std::optional<std::string>() : std::make_optional(prompt.reference_text));
    const std::string wrapped_text = "<|text_start|>" + combined + "<|text_end|>";

    prompt.style_token_ids = tokenizer_.encode(style_text);
    prompt.text_token_ids = tokenizer_.encode_with_nonverbal_tags(wrapped_text);
    return prompt;
}

int64_t OmniVoicePromptBuilder::frame_rate() const {
    return engine::models::omnivoice::frame_rate(*assets_);
}

int64_t OmniVoicePromptBuilder::estimate_target_tokens(
    std::string_view text,
    const std::optional<std::string> & ref_text,
    const std::optional<int64_t> & ref_audio_tokens,
    float speed) const {
    return engine::models::omnivoice::estimate_target_tokens(
        *assets_,
        trim_copy(text),
        ref_text,
        ref_audio_tokens,
        speed);
}

std::vector<std::string> OmniVoicePromptBuilder::chunk_text_punctuation(
    std::string_view text,
    int64_t chunk_len,
    std::optional<int64_t> min_chunk_len) const {
    return chunk_text_punctuation_impl(trim_copy(text), chunk_len, min_chunk_len);
}

}  // namespace engine::models::omnivoice

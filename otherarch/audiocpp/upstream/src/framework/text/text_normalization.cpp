#include "engine/framework/text/text_normalization.h"

#include "engine/framework/io/text.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::text {
namespace {

bool parse_non_negative_i64(std::string_view text, int64_t & out) {
    if (text.empty()) {
        return false;
    }
    int64_t value = 0;
    for (const char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const int digit = ch - '0';
        if (value > (std::numeric_limits<int64_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

std::string english_cardinal_under_1000(int value) {
    static const std::string units[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    };
    static const std::string tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    };

    if (value < 20) {
        return units[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const int ten = value / 10;
        const int one = value % 10;
        return one == 0 ? tens[static_cast<size_t>(ten)]
                        : tens[static_cast<size_t>(ten)] + " " + units[static_cast<size_t>(one)];
    }
    const int hundred = value / 100;
    const int rest = value % 100;
    return rest == 0 ? units[static_cast<size_t>(hundred)] + " hundred"
                     : units[static_cast<size_t>(hundred)] + " hundred " + english_cardinal_under_1000(rest);
}

std::string english_digits_individually(std::string_view digits) {
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out += english_cardinal_under_1000(digits[i] - '0');
    }
    return out;
}

std::string english_cardinal(int64_t value) {
    if (value < 0) {
        return "minus " + english_cardinal(-value);
    }
    if (value < 1000) {
        return english_cardinal_under_1000(static_cast<int>(value));
    }
    if (value < 1000000) {
        const int64_t thousands = value / 1000;
        const int64_t rest = value % 1000;
        return rest == 0 ? english_cardinal(thousands) + " thousand"
                         : english_cardinal(thousands) + " thousand " + english_cardinal(rest);
    }
    return std::to_string(value);
}

std::string english_cardinal_from_digits(std::string_view digits) {
    int64_t value = 0;
    if (!parse_non_negative_i64(digits, value)) {
        return english_digits_individually(digits);
    }
    return english_cardinal(value);
}

std::string english_year(int value) {
    if (value >= 1900 && value <= 1999) {
        const int rest = value - 1900;
        return rest == 0 ? "nineteen hundred" : "nineteen " + english_cardinal_under_1000(rest);
    }
    if (value >= 2000 && value <= 2099) {
        const int rest = value - 2000;
        if (rest == 0) {
            return "two thousand";
        }
        if (rest < 10) {
            return "two thousand " + english_cardinal_under_1000(rest);
        }
        return "twenty " + english_cardinal_under_1000(rest);
    }
    return english_cardinal(value);
}

std::string english_ordinal(int64_t value) {
    static const std::string ordinals[] = {
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
        "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    };
    static const std::string tens_ordinals[] = {
        "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
        "sixtieth", "seventieth", "eightieth", "ninetieth",
    };
    if (value < 20) {
        return ordinals[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const int64_t ten = value / 10;
        const int64_t one = value % 10;
        return one == 0 ? tens_ordinals[static_cast<size_t>(ten)]
                        : english_cardinal_under_1000(static_cast<int>(ten * 10)) + " " + english_ordinal(one);
    }
    if (value % 100 == 0) {
        return english_cardinal(value / 100) + " hundredth";
    }
    return english_cardinal(value - (value % 100)) + " " + english_ordinal(value % 100);
}

std::string english_ordinal_from_digits(std::string_view digits) {
    int64_t value = 0;
    if (!parse_non_negative_i64(digits, value)) {
        return english_digits_individually(digits);
    }
    return english_ordinal(value);
}

std::string english_fraction(std::string_view numerator_digits, std::string_view denominator_digits) {
    int64_t numerator = 0;
    int64_t denominator = 0;
    if (!parse_non_negative_i64(numerator_digits, numerator) ||
        !parse_non_negative_i64(denominator_digits, denominator)) {
        return english_digits_individually(numerator_digits) + " over " + english_digits_individually(denominator_digits);
    }
    if (denominator == 2) {
        return numerator == 1 ? "one half" : english_cardinal(numerator) + " halves";
    }
    std::string out = english_cardinal(numerator) + " " + english_ordinal(denominator);
    if (numerator != 1) {
        out.push_back('s');
    }
    return out;
}

std::string normalize_english_regex(
    std::string text,
    const std::regex & pattern,
    const std::function<std::string(const std::smatch &)> & replacement) {
    std::string out;
    auto begin = text.cbegin();
    auto end = text.cend();
    std::smatch match;
    while (std::regex_search(begin, end, match, pattern)) {
        out.append(begin, match[0].first);
        out += replacement(match);
        begin = match[0].second;
    }
    out.append(begin, end);
    return out;
}

std::string normalize_english_dates(std::string text) {
    const std::regex pattern(
        R"(\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?)",
        std::regex_constants::icase);
    return normalize_english_regex(std::move(text), pattern, [](const std::smatch & match) {
        int64_t day = 0;
        int64_t year = 0;
        std::string out = match[1].str() + " " +
            (parse_non_negative_i64(match[2].str(), day) ? english_ordinal(day) : english_digits_individually(match[2].str()));
        if (match[3].matched) {
            out += " ";
            out += parse_non_negative_i64(match[3].str(), year) ? english_year(static_cast<int>(year))
                                                                : english_digits_individually(match[3].str());
        }
        return out;
    });
}

std::string apply_index_tts_punctuation_map(std::string text) {
    const std::pair<const char *, const char *> replacements[] = {
        {"：", ","}, {"；", ","}, {";", ","}, {"，", ","}, {"。", "."}, {"！", "!"},
        {"？", "?"}, {"\n", " "}, {"·", "-"}, {"、", ","}, {"...", "…"}, {",,,", "…"},
        {"，，，", "…"}, {"……", "…"}, {"“", "'"}, {"”", "'"}, {"\"", "'"}, {"‘", "'"},
        {"’", "'"}, {"（", "'"}, {"）", "'"}, {"(", "'"}, {")", "'"}, {"《", "'"},
        {"》", "'"}, {"【", "'"}, {"】", "'"}, {"[", "'"}, {"]", "'"}, {"—", "-"},
        {"～", "-"}, {"~", "-"}, {"「", "'"}, {"」", "'"}, {":", ","},
    };
    for (const auto & [from, to] : replacements) {
        text = replace_all(std::move(text), from, to);
    }
    return text;
}

std::string uppercase_ascii(std::string text) {
    for (char & ch : text) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return text;
}

uint32_t decode_utf8_codepoint(std::string_view text, size_t offset, size_t & width) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(text[offset + i]); };
    const unsigned char first = byte(0);
    if ((first & 0x80U) == 0U) {
        width = 1;
        return first;
    }
    if ((first & 0xE0U) == 0xC0U && offset + 1 < text.size()) {
        width = 2;
        return ((first & 0x1FU) << 6U) | (byte(1) & 0x3FU);
    }
    if ((first & 0xF0U) == 0xE0U && offset + 2 < text.size()) {
        width = 3;
        return ((first & 0x0FU) << 12U) | ((byte(1) & 0x3FU) << 6U) | (byte(2) & 0x3FU);
    }
    if ((first & 0xF8U) == 0xF0U && offset + 3 < text.size()) {
        width = 4;
        return ((first & 0x07U) << 18U) | ((byte(1) & 0x3FU) << 12U) |
               ((byte(2) & 0x3FU) << 6U) | (byte(3) & 0x3FU);
    }
    throw std::runtime_error("text normalization received invalid UTF-8");
}

void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7FU) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
}

uint32_t halfwidth_katakana_to_fullwidth(uint32_t cp) {
    switch (cp) {
    case 0xFF66: return 0x30F2;
    case 0xFF67: return 0x30A1;
    case 0xFF68: return 0x30A3;
    case 0xFF69: return 0x30A5;
    case 0xFF6A: return 0x30A7;
    case 0xFF6B: return 0x30A9;
    case 0xFF6C: return 0x30E3;
    case 0xFF6D: return 0x30E5;
    case 0xFF6E: return 0x30E7;
    case 0xFF6F: return 0x30C3;
    case 0xFF70: return 0x30FC;
    case 0xFF71: return 0x30A2;
    case 0xFF72: return 0x30A4;
    case 0xFF73: return 0x30A6;
    case 0xFF74: return 0x30A8;
    case 0xFF75: return 0x30AA;
    case 0xFF76: return 0x30AB;
    case 0xFF77: return 0x30AD;
    case 0xFF78: return 0x30AF;
    case 0xFF79: return 0x30B1;
    case 0xFF7A: return 0x30B3;
    case 0xFF7B: return 0x30B5;
    case 0xFF7C: return 0x30B7;
    case 0xFF7D: return 0x30B9;
    case 0xFF7E: return 0x30BB;
    case 0xFF7F: return 0x30BD;
    case 0xFF80: return 0x30BF;
    case 0xFF81: return 0x30C1;
    case 0xFF82: return 0x30C4;
    case 0xFF83: return 0x30C6;
    case 0xFF84: return 0x30C8;
    case 0xFF85: return 0x30CA;
    case 0xFF86: return 0x30CB;
    case 0xFF87: return 0x30CC;
    case 0xFF88: return 0x30CD;
    case 0xFF89: return 0x30CE;
    case 0xFF8A: return 0x30CF;
    case 0xFF8B: return 0x30D2;
    case 0xFF8C: return 0x30D5;
    case 0xFF8D: return 0x30D8;
    case 0xFF8E: return 0x30DB;
    case 0xFF8F: return 0x30DE;
    case 0xFF90: return 0x30DF;
    case 0xFF91: return 0x30E0;
    case 0xFF92: return 0x30E1;
    case 0xFF93: return 0x30E2;
    case 0xFF94: return 0x30E4;
    case 0xFF95: return 0x30E6;
    case 0xFF96: return 0x30E8;
    case 0xFF97: return 0x30E9;
    case 0xFF98: return 0x30EA;
    case 0xFF99: return 0x30EB;
    case 0xFF9A: return 0x30EC;
    case 0xFF9B: return 0x30ED;
    case 0xFF9C: return 0x30EF;
    case 0xFF9D: return 0x30F3;
    default: return cp;
    }
}

uint32_t apply_katakana_voicing(uint32_t cp, uint32_t mark) {
    if (mark == 0x3099 || mark == 0xFF9E) {
        switch (cp) {
        case 0x30A6: return 0x30F4;
        case 0x30AB: return 0x30AC;
        case 0x30AD: return 0x30AE;
        case 0x30AF: return 0x30B0;
        case 0x30B1: return 0x30B2;
        case 0x30B3: return 0x30B4;
        case 0x30B5: return 0x30B6;
        case 0x30B7: return 0x30B8;
        case 0x30B9: return 0x30BA;
        case 0x30BB: return 0x30BC;
        case 0x30BD: return 0x30BE;
        case 0x30BF: return 0x30C0;
        case 0x30C1: return 0x30C2;
        case 0x30C4: return 0x30C5;
        case 0x30C6: return 0x30C7;
        case 0x30C8: return 0x30C9;
        case 0x30CF: return 0x30D0;
        case 0x30D2: return 0x30D3;
        case 0x30D5: return 0x30D6;
        case 0x30D8: return 0x30D9;
        case 0x30DB: return 0x30DC;
        case 0x30EF: return 0x30F7;
        case 0x30F0: return 0x30F8;
        case 0x30F1: return 0x30F9;
        case 0x30F2: return 0x30FA;
        default: return cp;
        }
    }
    if (mark == 0x309A || mark == 0xFF9F) {
        switch (cp) {
        case 0x30CF: return 0x30D1;
        case 0x30D2: return 0x30D4;
        case 0x30D5: return 0x30D7;
        case 0x30D8: return 0x30DA;
        case 0x30DB: return 0x30DD;
        default: return cp;
        }
    }
    return cp;
}

std::vector<uint32_t> japanese_codepoints(std::string_view text) {
    std::vector<uint32_t> out;
    for (size_t pos = 0; pos < text.size();) {
        size_t width = 0;
        uint32_t cp = decode_utf8_codepoint(text, pos, width);
        pos += width;
        if (cp >= 0x3041U && cp <= 0x3096U) {
            cp += 0x60U;
        } else if (cp == 0x309DU) {
            cp = 0x30FDU;
        } else if (cp == 0x309EU) {
            cp = 0x30FEU;
        } else {
            cp = halfwidth_katakana_to_fullwidth(cp);
        }
        out.push_back(cp);
    }
    for (size_t i = 1; i < out.size();) {
        if (out[i] == 0x3099U || out[i] == 0x309AU || out[i] == 0xFF9EU || out[i] == 0xFF9FU) {
            const uint32_t voiced = apply_katakana_voicing(out[i - 1], out[i]);
            if (voiced != out[i - 1]) {
                out[i - 1] = voiced;
                out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
        }
        ++i;
    }
    return out;
}

}  // namespace

std::string replace_all(std::string text, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return text;
    }
    for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size())) {
        text.replace(pos, from.size(), to);
    }
    return text;
}

std::string collapse_ascii_whitespace(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool previous_space = false;
    for (const char ch : text) {
        const bool space = ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
        if (space) {
            if (!previous_space) {
                out.push_back(' ');
            }
        } else {
            out.push_back(ch);
        }
        previous_space = space;
    }
    return engine::io::trim_ascii_whitespace(std::move(out));
}

std::string normalize_index_tts_punctuation(std::string text) {
    return apply_index_tts_punctuation_map(std::move(text));
}

std::string normalize_english_numbers(std::string text) {
    // URLs before anything else: "https://example.com" ->
    // "https comma slash slash example dot com" (wetext en verbalizes the
    // punctuation). The domain dots must not become decimal points later.
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(https?)://([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*))", std::regex_constants::icase),
        [](const std::smatch & match) {
            std::string domain = match[2].str();
            return match[1].str() + " comma slash slash " +
                   std::regex_replace(domain, std::regex(R"(\.)"), " dot ");
        });
    // Telephone runs: "135-4567-8900" -> digit-by-digit groups (wetext en).
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d{3})-(\d{4})-(\d{4})\b)"),
        [](const std::smatch & match) {
            return english_digits_individually(match[1].str()) + ", " +
                   english_digits_individually(match[2].str()) + ", " +
                   english_digits_individually(match[3].str());
        });
    // o'clock times: "3:00" -> "three,oh zero" (wetext en output, comma included).
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d{1,2}):00\b)"),
        [](const std::smatch & match) {
            return english_cardinal_from_digits(match[1].str()) + ",oh zero";
        });
    // Numeric dates: "2024/10/01" -> "the first of october twenty twenty four".
    static const char * const kMonths[] = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    };
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d{4})/(\d{1,2})/(\d{1,2})\b)"),
        [](const std::smatch & match) {
            int64_t year = 0;
            int64_t month = 0;
            int64_t day = 0;
            if (!parse_non_negative_i64(match[1].str(), year) ||
                !parse_non_negative_i64(match[2].str(), month) || month < 1 || month > 12 ||
                !parse_non_negative_i64(match[3].str(), day) || day < 1 || day > 31) {
                return match[0].str();
            }
            return "the " + english_ordinal(day) + " of " + kMonths[month - 1] + " " +
                   english_year(static_cast<int>(year));
        });
    // Currency: "$50" -> "fifty dollars".
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\$(\d+(?:\.\d+)?))"),
        [](const std::smatch & match) {
            std::string amount;
            const std::string value = match[1].str();
            const size_t dot = value.find('.');
            if (dot == std::string::npos) {
                amount = english_cardinal_from_digits(value);
            } else {
                amount = english_cardinal_from_digits(value.substr(0, dot)) + " point " +
                         english_digits_individually(value.substr(dot + 1));
            }
            return amount + (value == "1" ? " dollar" : " dollars");
        });
    // Negative numbers: "-5 degrees" -> "negative five degrees". The minus is
    // only a sign when not attached to a word/number ("GPT-5" stays intact).
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"((^|[^A-Za-z0-9])-(\d+))"),
        [](const std::smatch & match) {
            return match[1].str() + "negative " + english_cardinal_from_digits(match[2].str());
        });
    text = normalize_english_dates(std::move(text));
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+(?:\.\d+)?)%)"),
        [](const std::smatch & match) {
            return normalize_english_regex(
                match[1].str(),
                std::regex(R"((\d+)\.(\d+))"),
                [](const std::smatch & decimal) {
                    return english_cardinal_from_digits(decimal[1].str()) + " point " +
                           english_digits_individually(decimal[2].str());
                }) +
                " percent";
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)/(\d+)\b)"),
        [](const std::smatch & match) {
            return english_fraction(match[1].str(), match[2].str());
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)\.(\d+)\b)"),
        [](const std::smatch & match) {
            return english_cardinal_from_digits(match[1].str()) + " point " +
                   english_digits_individually(match[2].str());
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)(st|nd|rd|th)\b)", std::regex_constants::icase),
        [](const std::smatch & match) {
            return english_ordinal_from_digits(match[1].str());
        });
    // Split letter<->digit boundaries so digits attached to letters verbalize
    // like the official wetext English normalizer ("DS4" -> "DS four",
    // "R2D2" -> "R two D two", "4K" -> "four K"). Runs after dates/decimals/
    // ordinals so "1st", "2.5" and friends have already been expanded; the
    // standalone cardinal rule below then spells out each digit group.
    text = std::regex_replace(std::move(text), std::regex(R"(([A-Za-z])(\d))"), "$1 $2");
    text = std::regex_replace(std::move(text), std::regex(R"((\d)([A-Za-z]))"), "$1 $2");
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)\b)"),
        [](const std::smatch & match) {
            return english_cardinal_from_digits(match[1].str());
        });
    return text;
}

std::string normalize_english_text(std::string_view text, const EnglishTextNormalizationOptions & options) {
    std::string out = collapse_ascii_whitespace(text);
    if (options.expand_common_contractions) {
        out = std::regex_replace(
            out,
            std::regex(R"((what|where|who|which|how|t?here|it|s?he|that|this)'s)", std::regex_constants::icase),
            "$1 is");
    }
    if (options.spell_numbers) {
        out = normalize_english_numbers(std::move(out));
    }
    if (options.verbalize_symbols) {
        // Match the official wetext English normalizer: standalone ASCII
        // symbols are verbalized ("a_b" -> "a underscore b",
        // "C++" -> "C plus plus", "a=b" -> "a equal sign b"). Runs after
        // number spelling so "50%" has already become "fifty percent".
        const std::pair<const char *, const char *> symbol_words[] = {
            {"_", " underscore "},
            {"+", " plus "},
            {"=", " equal sign "},
            {"*", " asterisk "},
            {"&", " and "},
            {"#", " hash "},
            {"%", " percent "},
            {"|", " vertical bar "},
            {"~", " tilde "},
        };
        for (const auto & [from, to] : symbol_words) {
            out = replace_all(std::move(out), from, to);
        }
        out = collapse_ascii_whitespace(out);
    }
    if (options.index_tts_punctuation) {
        out = apply_index_tts_punctuation_map(std::move(out));
    }
    if (options.uppercase_ascii) {
        out = uppercase_ascii(std::move(out));
    }
    return out;
}

std::string normalize_japanese_text(std::string_view text) {
    std::string mapped;
    mapped.reserve(text.size());
    for (uint32_t cp : japanese_codepoints(collapse_ascii_whitespace(text))) {
        switch (cp) {
        case 0xFF01: cp = 0x0021; break;
        case 0xFF1F: cp = 0x003F; break;
        case 0xFF0C: cp = 0x3001; break;
        case 0xFF0E: cp = 0x3002; break;
        case 0xFF61: cp = 0x3002; break;
        case 0xFF64: cp = 0x3001; break;
        case 0xFF62: cp = 0x300C; break;
        case 0xFF63: cp = 0x300D; break;
        case 0x3000: cp = 0x0020; break;
        default: break;
        }
        append_utf8(mapped, cp);
    }
    mapped = replace_all(std::move(mapped), "……", "…");
    mapped = replace_all(std::move(mapped), "...", "…");
    mapped = replace_all(std::move(mapped), "..", "…");
    return collapse_ascii_whitespace(mapped);
}

}  // namespace engine::text

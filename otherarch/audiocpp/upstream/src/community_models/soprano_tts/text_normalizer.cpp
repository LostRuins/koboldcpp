#include "engine/community_models/soprano_tts/text_normalizer.h"

#include "engine/framework/text/text_normalization.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::soprano_tts {
namespace {

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

// Non-overlapping left-to-right replacement, matching Python re.sub
// semantics (the framework helper of the same shape, kept local so the
// normalizer stays self-contained).
std::string regex_expand_all(
    std::string text,
    const std::regex & pattern,
    const std::function<std::string(const std::smatch &)> & expand) {
    std::string out;
    auto begin = text.cbegin();
    const auto end = text.cend();
    std::smatch match;
    while (std::regex_search(begin, end, match, pattern)) {
        out.append(begin, match[0].first);
        out += expand(match);
        begin = match[0].second;
    }
    out.append(begin, end);
    return out;
}

std::string regex_replace_literal(
    std::string text,
    const std::regex & pattern,
    const std::string & replacement) {
    return regex_expand_all(std::move(text), pattern,
        [&replacement](const std::smatch &) { return replacement; });
}

size_t decode_utf8(const std::string & text, size_t pos, uint32_t & out) {
    const auto byte = [&](size_t offset) -> uint32_t {
        return static_cast<unsigned char>(text[pos + offset]);
    };
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    if (lead < 0x80U) {
        out = lead;
        return 1;
    }
    if ((lead & 0xE0U) == 0xC0U && pos + 1 < text.size() &&
        (byte(1) & 0xC0U) == 0x80U) {
        out = ((lead & 0x1FU) << 6U) | (byte(1) & 0x3FU);
        return 2;
    }
    if ((lead & 0xF0U) == 0xE0U && pos + 2 < text.size() &&
        (byte(1) & 0xC0U) == 0x80U && (byte(2) & 0xC0U) == 0x80U) {
        out = ((lead & 0x0FU) << 12U) | ((byte(1) & 0x3FU) << 6U) |
              (byte(2) & 0x3FU);
        return 3;
    }
    if ((lead & 0xF8U) == 0xF0U && pos + 3 < text.size() &&
        (byte(1) & 0xC0U) == 0x80U && (byte(2) & 0xC0U) == 0x80U &&
        (byte(3) & 0xC0U) == 0x80U) {
        out = ((lead & 0x07U) << 18U) | ((byte(1) & 0x3FU) << 12U) |
              ((byte(2) & 0x3FU) << 6U) | (byte(3) & 0x3FU);
        return 4;
    }
    out = 0xFFFDU;
    return 1;
}

// ---------------------------------------------------------------------------
// Stage 2: convert_to_ascii (reference: unidecode(text)).
// ---------------------------------------------------------------------------

#include "soprano_unidecode_table.inc"

std::string convert_to_ascii(const std::string & text) {
    const auto table_end = std::end(kUnidecodeTable);
    std::string out;
    out.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        uint32_t codepoint = 0;
        const size_t width = decode_utf8(text, pos, codepoint);
        pos += width;
        if (codepoint < 0x80U) {
            out.push_back(static_cast<char>(codepoint));
            continue;
        }
        const auto it = std::lower_bound(
            std::begin(kUnidecodeTable), table_end, codepoint,
            [](const auto & entry, uint32_t value) {
                return entry.codepoint < value;
            });
        if (it != table_end && it->codepoint == codepoint) {
            out += it->ascii;
        }
        // Unmapped non-ASCII codepoints are dropped here; the character
        // whitelist stage would strip them anyway.
    }
    return out;
}

// ---------------------------------------------------------------------------
// inflect.number_to_words equivalent (andword='', hyphenated compounds,
// comma group separators). Used by the number stage.
// ---------------------------------------------------------------------------

std::string cardinal_under_100(int value) {
    static const char * const units[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    };
    static const char * const tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    };
    if (value < 20) {
        return units[static_cast<size_t>(value)];
    }
    const int ten = value / 10;
    const int one = value % 10;
    return one == 0 ? std::string(tens[static_cast<size_t>(ten)])
                    : std::string(tens[static_cast<size_t>(ten)]) + "-" +
                          units[static_cast<size_t>(one)];
}

std::string cardinal_under_1000(int value) {
    if (value < 100) {
        return cardinal_under_100(value);
    }
    static const char * const units[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine",
    };
    const int hundred = value / 100;
    const int rest = value % 100;
    return rest == 0
        ? std::string(units[static_cast<size_t>(hundred)]) + " hundred"
        : std::string(units[static_cast<size_t>(hundred)]) + " hundred " +
              cardinal_under_100(rest);
}

std::string cardinal_words(int64_t value) {
    if (value < 0) {
        return "minus " + cardinal_words(-value);
    }
    if (value < 1000) {
        return cardinal_under_1000(static_cast<int>(value));
    }
    struct Scale {
        int64_t factor;
        const char * name;
    };
    static const Scale scales[] = {
        {1000000000LL, "billion"},
        {1000000LL, "million"},
        {1000LL, "thousand"},
    };
    std::string out;
    int64_t rest = value;
    for (const auto & scale : scales) {
        if (rest >= scale.factor) {
            const int64_t count = rest / scale.factor;
            rest %= scale.factor;
            if (!out.empty()) {
                out += ", ";
            }
            out += cardinal_words(count) + " " + scale.name;
        }
    }
    if (rest > 0) {
        if (!out.empty()) {
            out += ", ";
        }
        out += cardinal_under_1000(static_cast<int>(rest));
    }
    return out;
}

std::string ordinal_words(int64_t value) {
    static const char * const ordinals[] = {
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth",
    };
    static const char * const tens_ordinals[] = {
        "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
        "sixtieth", "seventieth", "eightieth", "ninetieth",
    };
    if (value < 20) {
        return ordinals[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const int64_t ten = value / 10;
        const int64_t one = value % 10;
        return one == 0
            ? std::string(tens_ordinals[static_cast<size_t>(ten)])
            : cardinal_under_100(static_cast<int>(ten * 10)) + "-" +
                  ordinals[static_cast<size_t>(one)];
    }
    if (value % 100 == 0) {
        if (value % 1000 == 0) {
            return cardinal_words(value / 1000) + " thousandth";
        }
        return cardinal_words(value / 100) + " hundredth";
    }
    return cardinal_words(value - value % 100) + " " +
           ordinal_words(value % 100);
}

// ---------------------------------------------------------------------------
// Stage 3: normalize_newlines (strip lines, ensure ".!?" endings).
// ---------------------------------------------------------------------------

std::string normalize_newlines(const std::string & text) {
    std::string out;
    size_t start = 0;
    for (;;) {
        const size_t end = text.find('\n', start);
        const size_t stop = end == std::string::npos ? text.size() : end;
        std::string line = text.substr(start, stop - start);
        size_t begin = 0;
        while (begin < line.size() &&
               std::isspace(static_cast<unsigned char>(line[begin])) != 0) {
            ++begin;
        }
        while (!line.empty() &&
               std::isspace(static_cast<unsigned char>(line.back())) != 0) {
            line.pop_back();
        }
        line = line.substr(begin);
        if (!line.empty()) {
            if (line.back() != '.' && line.back() != '!' && line.back() != '?') {
                line.push_back('.');
            }
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += line;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Stage 4: normalize_numbers.
// ---------------------------------------------------------------------------

// Port of _expand_time. Note: the reference h:m:s branch has a Python quirk
// where a bare `{minutes}` inside the f-string renders a one-element set, so
// non-"oh" minute values come out as "{'25'}" (braces + quotes). The braces
// are then turned into commas by the parentheses rule of normalize_special,
// reproducing e.g. "3:25:10" -> "three, 'twenty-five', ten".
std::string expand_time(const std::string & value) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        const size_t colon = value.find(':', start);
        parts.push_back(value.substr(
            start, colon == std::string::npos ? std::string::npos : colon - start));
        if (colon == std::string::npos) {
            break;
        }
        start = colon + 1;
    }

    const auto oh = [](const std::string & unit) {
        return "oh " + unit.substr(1);
    };
    if (parts.size() == 2) {
        const std::string & hours = parts[0];
        const std::string & minutes = parts[1];
        if (minutes == "00") {
            const int hour = std::stoi(hours);
            if (hour == 0) {
                return "0";
            }
            if (hour > 12) {
                return hours + " minutes";
            }
            return hours + " o'clock";
        }
        return hours + " " + (minutes[0] == '0' ? oh(minutes) : minutes);
    }
    const std::string & hours = parts[0];
    const std::string & minutes = parts[1];
    const std::string & seconds = parts[2];
    const auto seconds_part = [&]() -> std::string {
        if (seconds == "00") {
            return "";
        }
        return seconds[0] == '0' ? oh(seconds) : seconds;
    };
    if (std::stoi(hours) != 0) {
        const std::string minute_part = minutes == "00"
            ? "oh oh"
            : (minutes[0] == '0' ? oh(minutes) : "{'" + minutes + "'}");
        return hours + " " + minute_part + " " + seconds_part();
    }
    if (minutes != "00") {
        return minutes + " " + seconds_part();
    }
    return seconds;
}

// Port of _expand_dollars.
std::string expand_dollars(const std::string & amount) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        const size_t dot = amount.find('.', start);
        parts.push_back(amount.substr(
            start, dot == std::string::npos ? std::string::npos : dot - start));
        if (dot == std::string::npos) {
            break;
        }
        start = dot + 1;
    }
    if (parts.size() > 2) {
        return amount + " dollars";
    }
    const auto digits_only = [](const std::string & value) {
        std::string digits;
        for (const char ch : value) {
            if (ch >= '0' && ch <= '9') {
                digits.push_back(ch);
            }
        }
        return digits;
    };
    const int64_t dollars = parts[0].empty()
        ? 0
        : std::stoll(digits_only(parts[0]));
    const int64_t cents = parts.size() > 1 && !parts[1].empty()
        ? std::stoll(digits_only(parts[1]))
        : 0;
    if (dollars > 0 && cents > 0) {
        return cardinal_words(dollars) + (dollars == 1 ? " dollar" : " dollars") +
               ", " + cardinal_words(cents) + (cents == 1 ? " cent" : " cents");
    }
    if (dollars > 0) {
        return cardinal_words(dollars) + (dollars == 1 ? " dollar" : " dollars");
    }
    if (cents > 0) {
        return cardinal_words(cents) + (cents == 1 ? " cent" : " cents");
    }
    return "zero dollars";
}

// Port of _expand_number (the \d+ rule).
std::string expand_number(const std::string & digits) {
    // Python ints are arbitrary precision; clamp pathological runs (>18
    // digits) to individual digit spelling.
    if (digits.size() > 18) {
        std::string out;
        for (size_t i = 0; i < digits.size(); ++i) {
            if (i != 0) {
                out.push_back(' ');
            }
            out += cardinal_under_100(digits[i] - '0');
        }
        return out;
    }
    const int64_t num = std::stoll(digits);
    if (num > 1000 && num < 3000) {
        if (num == 2000) {
            return "two thousand";
        }
        if (num > 2000 && num < 2010) {
            return "two thousand " + cardinal_under_100(static_cast<int>(num % 100));
        }
        if (num % 100 == 0) {
            return cardinal_words(num / 100) + " hundred";
        }
        // inflect group=2 spelling with zero='oh': "1990" -> "nineteen ninety",
        // "1005" -> "ten oh five".
        const int leading = static_cast<int>(num / 100);
        const int trailing = static_cast<int>(num % 100);
        return cardinal_under_100(leading) + " " +
               (trailing < 10 ? "oh " + cardinal_under_100(trailing)
                              : cardinal_under_100(trailing));
    }
    return cardinal_words(num);
}

std::string normalize_numbers(const std::string & text) {
    static const std::regex num_prefix_re(R"(#\d)");
    static const std::regex num_suffix_re(R"(\b\d+(K|M|B|T)\b)",
        std::regex_constants::icase);
    static const std::regex comma_number_re(R"((\d[\d,]+\d))");
    static const std::regex date_re(
        R"((^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/]))");
    static const std::regex phone_number_re(R"((\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}))");
    static const std::regex time_re(R"((\d\d?:\d\d(?::\d\d)?))");
    static const std::regex pounds_re("\xC2\xA3" R"(([\d,]*\d+))");
    static const std::regex dollars_re(R"(\$([\d.,]*\d+))");
    static const std::regex decimal_number_re(R"((\d+(?:\.\d+)+))");
    static const std::regex multiply_re(R"((\d\s?\*\s?\d))");
    static const std::regex divide_re(R"((\d\s?/\s?\d))");
    static const std::regex add_re(R"((\d\s?\+\s?\d))");
    static const std::regex subtract_re(R"((\d?\s?-\s?\d))");
    static const std::regex fraction_re(R"((\d+(?:/\d+)+))");
    static const std::regex ordinal_re(R"(\d+(st|nd|rd|th))");
    static const std::regex letter_split_re(R"((\d[a-z]|[a-z]\d))",
        std::regex_constants::icase);
    static const std::regex number_re(R"(\d+)");

    std::string out = text;
    // "#5" -> "number 5" (only the first digit, like the reference).
    out = regex_expand_all(std::move(out), num_prefix_re, [](const std::smatch & m) {
        return std::string("number ") + m[0].str()[1];
    });
    // "100K" -> "100 thousand" (digits are spelled by the final \d+ rule).
    out = regex_expand_all(std::move(out), num_suffix_re, [](const std::smatch & m) {
        std::string value = m[0].str();
        const char suffix = static_cast<char>(
            std::toupper(static_cast<unsigned char>(value.back())));
        value.pop_back();
        const char * unit = suffix == 'K' ? "thousand"
            : suffix == 'M' ? "million"
            : suffix == 'B' ? "billion"
            : suffix == 'T' ? "trillion" : "";
        return value + " " + unit;
    });
    // "1,234" -> "1234".
    out = regex_expand_all(std::move(out), comma_number_re, [](const std::smatch & m) {
        std::string value = m[1].str();
        value.erase(std::remove(value.begin(), value.end(), ','), value.end());
        return value;
    });
    // "1/1/2025" -> "1 dash 1 dash 2025".
    out = regex_expand_all(std::move(out), date_re, [](const std::smatch & m) {
        std::string expanded;
        for (const char ch : m[2].str()) {
            if (ch == '.' || ch == '/' || ch == '-') {
                expanded += " dash ";
            } else {
                expanded.push_back(ch);
            }
        }
        return m[1].str() + expanded + m[3].str();
    });
    // "555-123-4567" -> "5 5 5, 1 2 3, 4 5 6 7".
    out = regex_expand_all(std::move(out), phone_number_re, [](const std::smatch & m) {
        std::string digits;
        for (const char ch : m[1].str()) {
            if (ch >= '0' && ch <= '9') {
                digits.push_back(ch);
            }
        }
        std::string grouped;
        for (size_t group = 0; group < 3; ++group) {
            if (group != 0) {
                grouped += ", ";
            }
            const size_t begin = group == 0 ? 0 : group == 1 ? 3 : 6;
            const size_t end = group == 0 ? 3 : group == 1 ? 6 : digits.size();
            for (size_t i = begin; i < end; ++i) {
                if (i != begin) {
                    grouped.push_back(' ');
                }
                grouped.push_back(digits[i]);
            }
        }
        return grouped;
    });
    // "3:00" -> "3 o'clock", "8:05" -> "8 oh 5", "3:25:10" -> "3 {'25'} 10".
    out = regex_expand_all(std::move(out), time_re, [](const std::smatch & m) {
        return expand_time(m[1].str());
    });
    // Kept for reference parity; the ASCII stage already folds "£" to "PS".
    out = regex_expand_all(std::move(out), pounds_re, [](const std::smatch & m) {
        return m[1].str() + " pounds";
    });
    // "$2.47" -> "two dollars, forty-seven cents".
    out = regex_expand_all(std::move(out), dollars_re, [](const std::smatch & m) {
        return expand_dollars(m[1].str());
    });
    // "1.5" -> "1 point 5" (digits of the fractional parts spelled separately).
    out = regex_expand_all(std::move(out), decimal_number_re, [](const std::smatch & m) {
        const std::string value = m[1].str();
        std::string expanded;
        size_t start = 0;
        for (;;) {
            const size_t dot = value.find('.', start);
            const std::string part = value.substr(
                start, dot == std::string::npos ? std::string::npos : dot - start);
            if (start != 0) {
                expanded += " point ";
                for (size_t i = 0; i < part.size(); ++i) {
                    if (i != 0) {
                        expanded.push_back(' ');
                    }
                    expanded.push_back(part[i]);
                }
            } else {
                expanded += part;
            }
            if (dot == std::string::npos) {
                break;
            }
            start = dot + 1;
        }
        return expanded;
    });
    // "2 * 3" -> "2  times  3" (the reference keeps surrounding spaces).
    const auto join_operator = [](const std::string & value, const char op,
                                  const char * word) {
        std::string expanded;
        size_t start = 0;
        for (;;) {
            const size_t hit = value.find(op, start);
            expanded += value.substr(
                start, hit == std::string::npos ? std::string::npos : hit - start);
            if (hit == std::string::npos) {
                break;
            }
            expanded += word;
            start = hit + 1;
        }
        return expanded;
    };
    out = regex_expand_all(std::move(out), multiply_re, [&](const std::smatch & m) {
        return join_operator(m[1].str(), '*', " times ");
    });
    out = regex_expand_all(std::move(out), divide_re, [&](const std::smatch & m) {
        return join_operator(m[1].str(), '/', " over ");
    });
    out = regex_expand_all(std::move(out), add_re, [&](const std::smatch & m) {
        return join_operator(m[1].str(), '+', " plus ");
    });
    out = regex_expand_all(std::move(out), subtract_re, [&](const std::smatch & m) {
        return join_operator(m[1].str(), '-', " minus ");
    });
    // "1/2" -> "1 over 2", "1/2/3" -> "1 slash 2 slash 3".
    out = regex_expand_all(std::move(out), fraction_re, [](const std::smatch & m) {
        const std::string value = m[1].str();
        std::vector<std::string> parts;
        size_t start = 0;
        for (;;) {
            const size_t slash = value.find('/', start);
            parts.push_back(value.substr(
                start, slash == std::string::npos ? std::string::npos : slash - start));
            if (slash == std::string::npos) {
                break;
            }
            start = slash + 1;
        }
        const char * separator = parts.size() == 2 ? " over " : " slash ";
        std::string expanded;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i != 0) {
                expanded += separator;
            }
            expanded += parts[i];
        }
        return expanded;
    });
    // "21st" -> "twenty-first".
    out = regex_expand_all(std::move(out), ordinal_re, [](const std::smatch & m) {
        const std::string value = m[0].str();
        return ordinal_words(std::stoll(value.substr(0, value.size() - 2)));
    });
    // "DS4" -> "DS 4", "R2D2" -> "R 2 D 2" (two passes, like the reference).
    for (int pass = 0; pass < 2; ++pass) {
        out = regex_expand_all(std::move(out), letter_split_re, [](const std::smatch & m) {
            const std::string pair = m[1].str();
            return std::string(1, pair[0]) + " " + pair[1];
        });
    }
    // Bare numbers (with the reference's 1000-3000 year handling).
    out = regex_expand_all(std::move(out), number_re, [](const std::smatch & m) {
        return expand_number(m[0].str());
    });
    return out;
}

// ---------------------------------------------------------------------------
// Stage 5: normalize_special (links, dashes, "A.B", brackets).
// ---------------------------------------------------------------------------

std::string normalize_special(const std::string & text) {
    static const std::regex link_header_re(R"((https?://))");
    static const std::regex dash_re(R"((. - .))");
    static const std::regex dot_re("([A-Z]\\.[A-Z])", std::regex_constants::icase);
    static const std::regex parentheses_re(R"([\(\[\{].*[\)\]\}](.|$))");
    static const std::regex open_bracket_re(R"([\(\[\{])");
    static const std::regex close_bracket_and_char_re(R"([\)\]\}][^$.!?,])");
    static const std::regex close_bracket_re(R"([\)\]\}])");

    std::string out = regex_replace_literal(
        text, link_header_re, "h t t p s colon slash slash ");
    // "x - y" -> "x, y".
    out = regex_expand_all(std::move(out), dash_re, [](const std::smatch & m) {
        const std::string value = m[1].str();
        return std::string(1, value[0]) + ", " + value[4];
    });
    // "e.g" -> "e dot g" (the trailing "." of "e.g." survives).
    out = regex_expand_all(std::move(out), dot_re, [](const std::smatch & m) {
        const std::string value = m[1].str();
        return std::string(1, value[0]) + " dot " + value[2];
    });
    // "[ok]" -> ", ok", "(note) next" -> ", note next" etc.
    out = regex_expand_all(std::move(out), parentheses_re, [](const std::smatch & m) {
        std::string value = m[0].str();
        value = std::regex_replace(value, open_bracket_re, ", ");
        value = std::regex_replace(value, close_bracket_and_char_re, ", ");
        value = std::regex_replace(value, close_bracket_re, "");
        return value;
    });
    return out;
}

// ---------------------------------------------------------------------------
// Stage 6: expand_abbreviations.
// ---------------------------------------------------------------------------

std::string expand_abbreviations(const std::string & text) {
    // Reference table 1: "\b<abbr>\." (IGNORECASE).
    static const std::pair<const char *, const char *> dot_abbreviations[] = {
        {"mrs", "misess"}, {"ms", "miss"}, {"mr", "mister"}, {"dr", "doctor"},
        {"st", "saint"}, {"co", "company"}, {"jr", "junior"}, {"maj", "major"},
        {"gen", "general"}, {"drs", "doctors"}, {"rev", "reverend"},
        {"lt", "lieutenant"}, {"hon", "honorable"}, {"sgt", "sergeant"},
        {"capt", "captain"}, {"esq", "esquire"}, {"ltd", "limited"},
        {"col", "colonel"}, {"ft", "fort"},
    };
    // Reference table 2: "\b<abbr>\b" (case-sensitive).
    static const std::pair<const char *, const char *> cased_abbreviations[] = {
        {"Hz", "hertz"}, {"kHz", "kilohertz"}, {"KBs", "kilobytes"},
        {"KB", "kilobyte"}, {"MBs", "megabytes"}, {"MB", "megabyte"},
        {"GBs", "gigabytes"}, {"GB", "gigabyte"}, {"TBs", "terabytes"},
        {"TB", "terabyte"}, {"APIs", "a p i's"}, {"API", "a p i"},
        {"CLIs", "c l i's"}, {"CLI", "c l i"}, {"CPUs", "c p u's"},
        {"CPU", "c p u"}, {"GPUs", "g p u's"}, {"GPU", "g p u"},
        {"Ave", "avenue"}, {"etc", "et cetera"}, {"Mon", "monday"},
        {"Tues", "tuesday"}, {"Wed", "wednesday"}, {"Thurs", "thursday"},
        {"Fri", "friday"}, {"Sat", "saturday"}, {"Jan", "january"},
        {"Feb", "february"}, {"Mar", "march"}, {"Apr", "april"},
        {"Aug", "august"}, {"Sept", "september"}, {"Oct", "october"},
        {"Nov", "november"}, {"Dec", "december"}, {"and/or", "and or"},
    };
    static const std::vector<std::pair<std::regex, std::string>> rules = [] {
        std::vector<std::pair<std::regex, std::string>> built;
        built.reserve(std::size(dot_abbreviations) + std::size(cased_abbreviations));
        for (const auto & entry : dot_abbreviations) {
            built.emplace_back(
                std::regex(std::string(R"(\b)") + entry.first + R"(\.)",
                           std::regex_constants::icase),
                entry.second);
        }
        for (const auto & entry : cased_abbreviations) {
            built.emplace_back(
                std::regex(std::string(R"(\b)") + entry.first + R"(\b)"),
                entry.second);
        }
        return built;
    }();
    std::string out = text;
    for (const auto & rule : rules) {
        out = regex_replace_literal(std::move(out), rule.first, rule.second);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Stage 7: normalize_mixedcase (CamelCase splitting).
// ---------------------------------------------------------------------------

std::string normalize_mixedcase(const std::string & text) {
    static const std::regex camelcase_re(R"(\b([A-Z][a-z]*)+\b)");
    static const std::regex component_re(R"([A-Z][a-z]*)");
    return regex_expand_all(text, camelcase_re, [](const std::smatch & match) {
        const std::string word = match[0].str();
        std::vector<std::string> components;
        for (std::sregex_iterator it(word.begin(), word.end(), component_re), end;
             it != end; ++it) {
            components.push_back(it->str());
        }
        if (components.size() == 1) {
            return word;  // Single capital word.
        }
        if (components.size() == word.size()) {
            return word;  // All uppercase.
        }
        if (components.size() == word.size() - 1 && word.back() == 's') {
            return word.substr(0, word.size() - 1) + "'s";  // Plural all-caps.
        }
        std::string out;
        for (size_t i = 0; i < components.size(); ++i) {
            if (i != 0) {
                out.push_back(' ');
            }
            out += components[i];
        }
        return out;
    });
}

// ---------------------------------------------------------------------------
// Stage 1 + 8: pre-unicode specials and special characters.
// ---------------------------------------------------------------------------

std::string expand_preunicode_special_characters(const std::string & text) {
    // Reference table: em-dash -> " - " (the en-dash is folded to "-" by the
    // ASCII stage instead).
    return engine::text::replace_all(text, "\xE2\x80\x94", " - ");
}

std::string expand_special_characters(const std::string & text) {
    static const std::vector<std::pair<std::regex, std::string>> rules = {
        {std::regex("@"), " at "},
        {std::regex("&"), " and "},
        {std::regex("%"), " percent "},
        {std::regex(":"), "."},
        {std::regex(";"), ","},
        {std::regex(R"(\+)"), " plus "},
        {std::regex(R"(\\)"), " backslash "},
        {std::regex("~"), " about "},
        {std::regex("(^| )<3"), " heart "},
        {std::regex("<="), " less than or equal to "},
        {std::regex(">="), " greater than or equal to "},
        {std::regex("<"), " less than "},
        {std::regex(">"), " greater than "},
        {std::regex("="), " equals "},
        {std::regex("/"), " slash "},
        {std::regex("_"), " "},
        {std::regex(R"(\*)"), " "},
    };
    std::string out = text;
    for (const auto & rule : rules) {
        out = regex_replace_literal(std::move(out), rule.first, rule.second);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Stages 9-13 (moved from tokenizer_text.cpp): lowercase, whitelist, collapse,
// dedup, triple letters.
// ---------------------------------------------------------------------------

constexpr char kEllipsisPlaceholder = '\x01';

// Net whitelist of the reference `remove_unknown_characters()`: the first
// pass keeps [A-Za-z !$%&'*+,-./0-9<>?_] and the second drops "<>/_+".
// Everything else (brackets, quotes, colon, emoji, non-ASCII diacritics, ...)
// is stripped, exactly like the reference.
bool is_allowed_character(char ch) {
    const auto c = static_cast<unsigned char>(ch);
    return c == ' ' ||
        (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        c == '!' || c == '$' || c == '%' || c == '&' || c == '\'' ||
        c == '*' || c == ',' || c == '-' || c == '.' || c == '?';
}

std::string remove_unknown_characters(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        if (is_allowed_character(ch)) {
            out.push_back(ch);
        }
    }
    return out;
}

// `collapse_whitespace()`: collapse whitespace runs to one space, trim the
// ends, and drop a space directly before sentence punctuation
// ("word ." -> "word.").
std::string collapse_whitespace(const std::string & text) {
    std::string collapsed;
    collapsed.reserve(text.size());
    bool prev_space = false;
    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!prev_space && !collapsed.empty()) {
                collapsed.push_back(' ');
            }
            prev_space = true;
        } else {
            collapsed.push_back(ch);
            prev_space = false;
        }
    }
    while (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
    }
    std::string out;
    out.reserve(collapsed.size());
    for (size_t i = 0; i < collapsed.size(); ++i) {
        if (collapsed[i] == ' ' && i + 1 < collapsed.size()) {
            const char next = collapsed[i + 1];
            if (next == '.' || next == '?' || next == '!' || next == ',') {
                continue;
            }
        }
        out.push_back(collapsed[i]);
    }
    return out;
}

// `dedup_punctuation()`: collapse punctuation runs onto a single mark while
// preserving a true ellipsis ("..." stays "..."; "..", ".," and ".?!" collapse
// onto the strongest mark of the run).
std::string dedup_punctuation(const std::string & text) {
    // 1) Protect ellipses (3+ dots) with a placeholder, matching the
    //    reference "\\.\.\.+" -> "[ELLIPSIS]" round-trip.
    std::string protected_text;
    protected_text.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '.') {
            size_t j = i;
            while (j < text.size() && text[j] == '.') {
                ++j;
            }
            if (j - i >= 3) {
                protected_text.push_back(kEllipsisPlaceholder);
            } else {
                protected_text.append(text, i, j - i);
            }
            i = j;
        } else {
            protected_text.push_back(text[i]);
            ++i;
        }
    }

    // 2)-5) The reference collapses runs with the regexes ",+" -> "," and
    //    "[class]*<mark>[class]*" -> "<mark>" for the nested classes
    //    {.,} with '.', {.,!} with '!' and {.,!?} with '?'. Each maximal run
    //    over the class set becomes the mark when it contains it.
    const auto collapse_runs = [](const std::string & input, char mark,
                                  auto in_class) {
        std::string out;
        out.reserve(input.size());
        for (size_t i = 0; i < input.size();) {
            if (!in_class(input[i])) {
                out.push_back(input[i]);
                ++i;
                continue;
            }
            size_t j = i;
            bool has_mark = false;
            while (j < input.size() && in_class(input[j])) {
                has_mark = has_mark || input[j] == mark;
                ++j;
            }
            if (has_mark) {
                out.push_back(mark);
            } else {
                out.append(input, i, j - i);
            }
            i = j;
        }
        return out;
    };

    // 2) ",+" -> ","
    std::string current;
    current.reserve(protected_text.size());
    for (size_t i = 0; i < protected_text.size();) {
        if (protected_text[i] != ',') {
            current.push_back(protected_text[i]);
            ++i;
            continue;
        }
        size_t j = i;
        while (j < protected_text.size() && protected_text[j] == ',') {
            ++j;
        }
        current.push_back(',');
        i = j;
    }
    // 3) "[.,]*.[.,]*" -> "."
    current = collapse_runs(current, '.', [](char ch) {
        return ch == '.' || ch == ',';
    });
    // 4) "[.,!]*![.,!]*" -> "!"
    current = collapse_runs(current, '!', [](char ch) {
        return ch == '.' || ch == ',' || ch == '!';
    });
    // 5) "[.,!?]*?[.,!?]*" -> "?"
    current = collapse_runs(current, '?', [](char ch) {
        return ch == '.' || ch == ',' || ch == '!' || ch == '?';
    });

    // 6) Restore the ellipses.
    std::string out;
    out.reserve(current.size());
    for (const char ch : current) {
        if (ch == kEllipsisPlaceholder) {
            out.append("...");
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

// `collapse_triple_letters()`: "goooood" -> "good" (keep at most two of any
// repeated word character).
std::string collapse_triple_letters(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        while (j < text.size() && text[j] == text[i]) {
            ++j;
        }
        const auto c = static_cast<unsigned char>(text[i]);
        const bool word_char = std::isalnum(c) != 0 || text[i] == '_';
        if (word_char && j - i >= 3) {
            out.append(2, text[i]);
        } else {
            out.append(text, i, j - i);
        }
        i = j;
    }
    return out;
}

}  // namespace

std::string clean_soprano_text(const std::string & text) {
    std::string out = expand_preunicode_special_characters(text);
    out = convert_to_ascii(std::move(out));
    out = normalize_newlines(std::move(out));
    out = normalize_numbers(std::move(out));
    out = normalize_special(std::move(out));
    out = expand_abbreviations(std::move(out));
    out = normalize_mixedcase(std::move(out));
    out = expand_special_characters(std::move(out));
    for (char & ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    out = remove_unknown_characters(std::move(out));
    out = collapse_whitespace(std::move(out));
    out = dedup_punctuation(std::move(out));
    out = collapse_triple_letters(std::move(out));
    return out;
}

}  // namespace engine::community_models::soprano_tts

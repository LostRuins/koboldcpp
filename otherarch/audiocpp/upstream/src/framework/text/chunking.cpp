#include "engine/framework/text/chunking.h"

#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::text {
namespace {

struct Utf8Span {
    size_t start = 0;
    size_t end = 0;
    std::string_view text;
};

struct WordRange {
    size_t span_start = 0;
    size_t span_end = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
    bool sentence_break = false;
    bool clause_break = false;
};

struct TextUnit {
    size_t span_start = 0;
    size_t span_end = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
    bool tag = false;
    bool sentence_break = false;
};

bool is_ascii_space(std::string_view token) noexcept {
    return token.size() == 1 && std::isspace(static_cast<unsigned char>(token.front())) != 0;
}

bool is_ascii_line_break(std::string_view token) noexcept {
    return token == "\n" || token == "\r";
}

bool is_horizontal_ascii_space(std::string_view token) noexcept {
    return token == " " || token == "\t" || token == "\f" || token == "\v";
}

bool is_sentence_break(std::string_view token) {
    return token == "." || token == "!" || token == "?" ||
           token == u8"。" || token == u8"！" || token == u8"？";
}

// Closing quotes and brackets that end a sentence together with the stop before
// them: `money."`, `ready?)`, `»`, `」`. Tag-aware units end at the stop, so
// without this the closer opened the next sentence and chunk.
bool is_sentence_closer(std::string_view token) {
    return token == "\"" || token == "'" || token == ")" ||
           token == u8"”" || token == u8"’" || token == u8"»" ||
           token == u8"」" || token == u8"』" || token == u8"）";
}

// True when spans [start, end) end in a sentence stop, optionally followed by
// closing quotes or brackets.
bool ends_sentence(const std::vector<Utf8Span> & spans, size_t start, size_t end) {
    while (end > start && is_sentence_closer(spans[end - 1].text)) {
        --end;
    }
    return end > start && is_sentence_break(spans[end - 1].text);
}

bool is_clause_break(std::string_view token) {
    return token == "," || token == ";" || token == ":" ||
           token == u8"，" || token == u8"、" || token == u8"；" || token == u8"：";
}

bool is_cjk_punctuation_delimiter(std::string_view token) noexcept {
    // Full-width CJK sentence/clause punctuation (。！？，、；：) separates words
    // even without ASCII spaces — CJK text has no inter-word spaces, so without
    // this a whole paragraph parses as one word and Default-mode chunking can
    // never split it at the codepoint budget (only TagAware/Japanese could).
    // ASCII punctuation stays attached to its word so space-delimited Latin
    // text is unaffected.
    const auto leading = static_cast<unsigned char>(token.front());
    return leading >= 0x80 && (is_sentence_break(token) || is_clause_break(token));
}

bool is_tag_open(std::string_view token) {
    return token == "[" || token == "<" || token == "(";
}

// End of the closing quotes and brackets that belong to the sentence stop just
// before `pos`. `"`, `'` and `’` also open words (`'Tis`, `’90s`, a quote right
// after a stop), so they belong to the stop only when the run of closers ends at
// whitespace, the end of the text or a tag, or when an unambiguous closer
// follows them.
size_t sentence_closers_end(const std::vector<Utf8Span> & spans, size_t pos) {
    size_t end = pos;
    size_t run_end = pos;
    while (run_end < spans.size() && is_sentence_closer(spans[run_end].text)) {
        const auto token = spans[run_end].text;
        ++run_end;
        if (token != "\"" && token != "'" && token != u8"’") {
            end = run_end;
        }
    }
    if (run_end == spans.size() || is_ascii_space(spans[run_end].text) || is_tag_open(spans[run_end].text)) {
        end = run_end;
    }
    return end;
}

bool is_tag_close(std::string_view token, std::string_view open) {
    return (open == "[" && token == "]") || (open == "<" && token == ">") ||
           (open == "(" && token == ")");
}

std::vector<Utf8Span> split_utf8_spans(std::string_view text, std::string_view label) {
    std::vector<Utf8Span> spans;
    spans.reserve(utf8_codepoint_count(text, label));
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        if (ch <= 0x7FU) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error(std::string(label) + " contains invalid UTF-8");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error(std::string(label) + " contains truncated UTF-8");
        }
        for (size_t i = 1; i < width; ++i) {
            if (!is_utf8_continuation(static_cast<unsigned char>(text[pos + i]))) {
                throw std::runtime_error(std::string(label) + " contains invalid UTF-8 continuation byte");
            }
        }
        spans.push_back({pos, pos + width, text.substr(pos, width)});
        pos += width;
    }
    return spans;
}

std::vector<WordRange> split_word_ranges(const std::vector<Utf8Span> & spans) {
    std::vector<WordRange> words;
    size_t span_pos = 0;
    while (span_pos < spans.size()) {
        while (span_pos < spans.size() && is_ascii_space(spans[span_pos].text)) {
            ++span_pos;
        }
        if (span_pos >= spans.size()) {
            break;
        }
        // A CJK punctuation token forms a word of its own (attached to nothing),
        // so a subsequent run never absorbs it and boundaries can land on it.
        if (is_cjk_punctuation_delimiter(spans[span_pos].text)) {
            words.push_back({
                span_pos,
                span_pos + 1,
                spans[span_pos].start,
                spans[span_pos].end,
                is_sentence_break(spans[span_pos].text),
                is_clause_break(spans[span_pos].text),
            });
            span_pos += 1;
            continue;
        }
        const size_t word_start = span_pos;
        size_t word_end = span_pos + 1;
        while (word_end < spans.size() &&
               !is_ascii_space(spans[word_end].text) &&
               !is_cjk_punctuation_delimiter(spans[word_end].text)) {
            ++word_end;
        }
        const auto last = spans[word_end - 1].text;
        words.push_back({
            word_start,
            word_end,
            spans[word_start].start,
            spans[word_end - 1].end,
            is_sentence_break(last),
            is_clause_break(last),
        });
        span_pos = word_end;
    }
    return words;
}

std::vector<TextUnit> split_tag_aware_units(const std::vector<Utf8Span> & spans) {
    std::vector<TextUnit> units;
    size_t span_pos = 0;
    while (span_pos < spans.size()) {
        while (span_pos < spans.size() && is_ascii_space(spans[span_pos].text)) {
            ++span_pos;
        }
        if (span_pos >= spans.size()) {
            break;
        }

        const size_t unit_start = span_pos;
        bool tag = false;
        if (is_tag_open(spans[span_pos].text)) {
            const auto open = spans[span_pos].text;
            ++span_pos;
            while (span_pos < spans.size() && !is_tag_close(spans[span_pos].text, open)) {
                ++span_pos;
            }
            if (span_pos < spans.size()) {
                ++span_pos;
                tag = true;
            } else {
                span_pos = unit_start + 1;
                while (span_pos < spans.size() &&
                       !is_ascii_space(spans[span_pos].text) &&
                       !is_tag_open(spans[span_pos].text)) {
                    ++span_pos;
                }
            }
        } else {
            ++span_pos;
            while (span_pos < spans.size() &&
                   !is_ascii_space(spans[span_pos].text) &&
                   !is_tag_open(spans[span_pos].text) &&
                   !is_sentence_break(spans[span_pos - 1].text)) {
                ++span_pos;
            }
            if (is_sentence_break(spans[span_pos - 1].text)) {
                span_pos = sentence_closers_end(spans, span_pos);
            }
        }

        units.push_back({
            unit_start,
            span_pos,
            spans[unit_start].start,
            spans[span_pos - 1].end,
            tag,
            !tag && ends_sentence(spans, unit_start, span_pos),
        });
    }
    return units;
}

std::string join_units(
    const std::string & text,
    const std::vector<TextUnit> & units,
    size_t start,
    size_t end) {
    if (start >= end) {
        return {};
    }
    return engine::io::trim_ascii_whitespace(
        text.substr(units[start].byte_start, units[end - 1].byte_end - units[start].byte_start));
}

int64_t unit_range_codepoints(const std::vector<TextUnit> & units, size_t start, size_t end) {
    if (start >= end) {
        return 0;
    }
    return static_cast<int64_t>(units[end - 1].span_end - units[start].span_start);
}

void append_piece_chunks(
    const std::string & text,
    const std::vector<TextUnit> & units,
    size_t start,
    size_t end,
    int64_t codepoint_budget,
    std::vector<std::string> & chunks) {
    size_t unit_start = start;
    while (unit_start < end) {
        size_t hard_end = unit_start;
        while (hard_end < end && unit_range_codepoints(units, unit_start, hard_end + 1) <= codepoint_budget) {
            ++hard_end;
        }
        if (hard_end == unit_start) {
            hard_end = unit_start + 1;
        }
        auto chunk = join_units(text, units, unit_start, hard_end);
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        unit_start = hard_end;
    }
}

std::vector<std::string> split_tag_aware_body(
    const std::string & text,
    int64_t codepoint_budget) {
    const auto spans = split_utf8_spans(text, "tag-aware text chunk");
    if (static_cast<int64_t>(spans.size()) <= codepoint_budget) {
        return {text};
    }
    const auto units = split_tag_aware_units(spans);
    if (units.empty()) {
        return {};
    }

    std::vector<std::pair<size_t, size_t>> sentences;
    size_t sentence_start = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        if (units[i].sentence_break) {
            sentences.emplace_back(sentence_start, i + 1);
            sentence_start = i + 1;
        }
    }
    if (sentence_start < units.size()) {
        sentences.emplace_back(sentence_start, units.size());
    }

    std::vector<std::string> chunks;
    size_t sentence_index = 0;
    while (sentence_index < sentences.size()) {
        const auto [sentence_start_unit, sentence_end_unit] = sentences[sentence_index];
        if (unit_range_codepoints(units, sentence_start_unit, sentence_end_unit) > codepoint_budget) {
            append_piece_chunks(text, units, sentence_start_unit, sentence_end_unit, codepoint_budget, chunks);
            ++sentence_index;
            continue;
        }

        size_t chunk_sentence_end = sentence_index;
        while (chunk_sentence_end < sentences.size()) {
            const size_t chunk_start_unit = sentences[sentence_index].first;
            const size_t candidate_end_unit = sentences[chunk_sentence_end].second;
            if (unit_range_codepoints(units, chunk_start_unit, candidate_end_unit) > codepoint_budget) {
                break;
            }
            ++chunk_sentence_end;
        }
        if (chunk_sentence_end == sentence_index) {
            chunk_sentence_end = sentence_index + 1;
        }
        auto chunk = join_units(
            text,
            units,
            sentences[sentence_index].first,
            sentences[chunk_sentence_end - 1].second);
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        sentence_index = chunk_sentence_end;
    }
    return chunks;
}

std::optional<std::pair<std::string, std::string>> split_leading_parenthetical_control(const std::string & text) {
    if (text.empty() || text.front() != '(') {
        return std::nullopt;
    }
    const size_t close = text.find(')');
    if (close == std::string::npos || close + 1 >= text.size()) {
        return std::nullopt;
    }
    const std::string prefix = engine::io::trim_ascii_whitespace(text.substr(0, close + 1));
    const std::string body = engine::io::trim_ascii_whitespace(text.substr(close + 1));
    if (prefix.empty() || body.empty()) {
        return std::nullopt;
    }
    return std::make_pair(prefix, body);
}

std::vector<std::string> split_text_chunks_default(
    std::string_view text,
    int64_t codepoint_budget) {
    if (codepoint_budget <= 0) {
        throw std::runtime_error("text chunk budget must be positive");
    }
    const std::string trimmed = engine::io::trim_ascii_whitespace(std::string(text));
    if (trimmed.empty()) {
        return {};
    }
    const auto spans = split_utf8_spans(trimmed, "text chunk");
    if (static_cast<int64_t>(spans.size()) <= codepoint_budget) {
        return {trimmed};
    }

    const auto words = split_word_ranges(spans);
    if (words.empty()) {
        return {};
    }

    std::vector<std::string> chunks;
    size_t word_start = 0;
    while (word_start < words.size()) {
        size_t hard_end = word_start;
        while (hard_end < words.size()) {
            const size_t chunk_span_start = words[word_start].span_start;
            const size_t candidate_span_end = words[hard_end].span_end;
            const auto chunk_codepoints = static_cast<int64_t>(candidate_span_end - chunk_span_start);
            if (chunk_codepoints > codepoint_budget) {
                break;
            }
            ++hard_end;
        }

        if (hard_end == word_start) {
            hard_end = word_start + 1;
        }

        size_t chunk_end = hard_end;
        if (hard_end < words.size() && hard_end > word_start + 1) {
            for (size_t i = hard_end; i > word_start + 1; --i) {
                if (words[i - 1].sentence_break) {
                    chunk_end = i;
                    break;
                }
            }
            if (chunk_end == hard_end) {
                for (size_t i = hard_end; i > word_start + 1; --i) {
                    if (words[i - 1].clause_break) {
                        chunk_end = i;
                        break;
                    }
                }
            }
        }

        const size_t byte_start = words[word_start].byte_start;
        const size_t byte_end = words[chunk_end - 1].byte_end;
        auto chunk = engine::io::trim_ascii_whitespace(trimmed.substr(byte_start, byte_end - byte_start));
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        word_start = chunk_end;
    }
    return chunks;
}

std::vector<std::string> split_text_chunks_tag_aware(
    std::string_view text,
    int64_t codepoint_budget) {
    if (codepoint_budget <= 0) {
        throw std::runtime_error("text chunk budget must be positive");
    }
    const std::string trimmed = engine::io::trim_ascii_whitespace(std::string(text));
    if (trimmed.empty()) {
        return {};
    }
    const auto parenthetical = split_leading_parenthetical_control(trimmed);
    if (!parenthetical.has_value()) {
        return split_tag_aware_body(trimmed, codepoint_budget);
    }

    const auto prefix_codepoints = static_cast<int64_t>(
        utf8_codepoint_count(parenthetical->first, "tag-aware leading text chunk tag"));
    const int64_t body_budget = std::max<int64_t>(1, codepoint_budget - prefix_codepoints - 1);
    auto chunks = split_tag_aware_body(parenthetical->second, body_budget);
    for (auto & chunk : chunks) {
        chunk = parenthetical->first + " " + chunk;
    }
    return chunks;
}

std::vector<std::string> split_text_chunks_japanese(
    std::string_view text,
    int64_t codepoint_budget) {
    if (codepoint_budget <= 0) {
        throw std::runtime_error("text chunk budget must be positive");
    }
    const std::string trimmed = engine::io::trim_ascii_whitespace(std::string(text));
    if (trimmed.empty()) {
        return {};
    }
    const auto spans = split_utf8_spans(trimmed, "Japanese text chunk");
    if (static_cast<int64_t>(spans.size()) <= codepoint_budget) {
        return {trimmed};
    }

    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < spans.size()) {
        while (start < spans.size() && is_ascii_space(spans[start].text)) {
            ++start;
        }
        if (start >= spans.size()) {
            break;
        }

        const size_t hard_end = std::min(
            spans.size(),
            start + static_cast<size_t>(codepoint_budget));
        size_t end = hard_end;
        if (hard_end < spans.size()) {
            for (size_t i = hard_end; i > start + 1; --i) {
                if (is_sentence_break(spans[i - 1].text)) {
                    end = i;
                    break;
                }
            }
            if (end == hard_end) {
                for (size_t i = hard_end; i > start + 1; --i) {
                    if (is_clause_break(spans[i - 1].text)) {
                        end = i;
                        break;
                    }
                }
            }
        }

        auto chunk = engine::io::trim_ascii_whitespace(
            trimmed.substr(spans[start].start, spans[end - 1].end - spans[start].start));
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        start = end;
    }
    return chunks;
}

void append_endline_chunk(
    const std::string & chunk,
    int64_t codepoint_budget,
    std::vector<std::string> & chunks) {
    if (chunk.empty()) {
        return;
    }
    if (static_cast<int64_t>(utf8_codepoint_count(chunk, "endline text chunk")) <= codepoint_budget) {
        chunks.push_back(chunk);
        return;
    }
    auto pieces = split_text_chunks_japanese(chunk, codepoint_budget);
    chunks.insert(
        chunks.end(),
        std::make_move_iterator(pieces.begin()),
        std::make_move_iterator(pieces.end()));
}

std::vector<std::string> split_text_chunks_endline(
    std::string_view text,
    int64_t codepoint_budget) {
    if (codepoint_budget <= 0) {
        throw std::runtime_error("text chunk budget must be positive");
    }
    const std::string trimmed = engine::io::trim_ascii_whitespace(std::string(text));
    if (trimmed.empty()) {
        return {};
    }
    const auto spans = split_utf8_spans(trimmed, "endline text chunk");

    std::vector<std::string> chunks;
    size_t chunk_start = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (!is_sentence_break(spans[i].text)) {
            continue;
        }

        size_t next = i + 1;
        while (next < spans.size() && is_horizontal_ascii_space(spans[next].text)) {
            ++next;
        }
        const bool followed_by_line_end =
            next >= spans.size() || is_ascii_line_break(spans[next].text);
        if (!followed_by_line_end) {
            continue;
        }

        auto chunk = engine::io::trim_ascii_whitespace(
            trimmed.substr(spans[chunk_start].start, spans[i].end - spans[chunk_start].start));
        append_endline_chunk(chunk, codepoint_budget, chunks);
        chunk_start = i + 1;
    }

    if (chunk_start < spans.size()) {
        auto tail = engine::io::trim_ascii_whitespace(
            trimmed.substr(spans[chunk_start].start, spans.back().end - spans[chunk_start].start));
        append_endline_chunk(tail, codepoint_budget, chunks);
    }
    return chunks;
}

}  // namespace

std::string_view text_chunk_mode_name(TextChunkMode mode) {
    switch (mode) {
    case TextChunkMode::Default:
        return "default";
    case TextChunkMode::TagAware:
        return "tag_aware";
    case TextChunkMode::Japanese:
        return "japanese";
    case TextChunkMode::Endline:
        return "endline";
    }
    return "unknown";
}

std::vector<std::string> split_text_chunks(
    std::string_view text,
    int64_t codepoint_budget) {
    return split_text_chunks_default(text, codepoint_budget);
}

std::vector<std::string> split_text_chunks(
    std::string_view text,
    int64_t codepoint_budget,
    TextChunkMode mode) {
    if (mode == TextChunkMode::TagAware) {
        return split_text_chunks_tag_aware(text, codepoint_budget);
    }
    if (mode == TextChunkMode::Japanese) {
        return split_text_chunks_japanese(text, codepoint_budget);
    }
    if (mode == TextChunkMode::Endline) {
        return split_text_chunks_endline(text, codepoint_budget);
    }
    return split_text_chunks_default(text, codepoint_budget);
}

std::optional<int64_t> parse_text_chunk_size_override(
    const std::unordered_map<std::string, std::string> & options) {
    const auto match = runtime::find_option_match(options, {"text_chunk_size", "chunk_size"});
    if (!match.has_value()) {
        return std::nullopt;
    }
    size_t consumed = 0;
    const int64_t value = std::stoll(match->value, &consumed);
    if (consumed != match->value.size()) {
        throw std::runtime_error(std::string(match->key) + " must be an integer");
    }
    if (value <= 0) {
        throw std::runtime_error(std::string(match->key) + " must be positive");
    }
    return value;
}

std::optional<TextChunkMode> parse_text_chunk_mode_override(
    const std::unordered_map<std::string, std::string> & options) {
    const auto match = runtime::find_option_match(options, {"text_chunk_mode", "chunk_mode"});
    if (!match.has_value()) {
        return std::nullopt;
    }
    const std::string & value = match->value;
    if (value == "default" || value == "word_budget") {
        return TextChunkMode::Default;
    }
    if (value == "tag_aware" || value == "tag-aware" || value == "tagaware") {
        return TextChunkMode::TagAware;
    }
    if (value == "japanese" || value == "ja") {
        return TextChunkMode::Japanese;
    }
    if (value == "endline") {
        return TextChunkMode::Endline;
    }
    throw std::runtime_error(
        std::string(match->key) +
        " must be one of default (or alias: word_budget), tag_aware, japanese, endline");
}

}  // namespace engine::text

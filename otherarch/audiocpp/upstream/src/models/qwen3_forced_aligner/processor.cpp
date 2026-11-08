#include "engine/models/qwen3_forced_aligner/processor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace engine::models::qwen3_forced_aligner {
namespace {

struct Utf8Char {
    uint32_t codepoint = 0;
    std::string text;
};

std::vector<Utf8Char> utf8_chars(const std::string & text) {
    std::vector<Utf8Char> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        size_t len = 1;
        if ((ch & 0x80U) == 0) {
            codepoint = ch;
        } else if ((ch & 0xE0U) == 0xC0U) {
            len = 2;
            codepoint = ch & 0x1FU;
        } else if ((ch & 0xF0U) == 0xE0U) {
            len = 3;
            codepoint = ch & 0x0FU;
        } else if ((ch & 0xF8U) == 0xF0U) {
            len = 4;
            codepoint = ch & 0x07U;
        } else {
            throw std::runtime_error("Qwen3 forced aligner encountered invalid UTF-8");
        }
        if (i + len > text.size()) {
            throw std::runtime_error("Qwen3 forced aligner encountered truncated UTF-8");
        }
        for (size_t j = 1; j < len; ++j) {
            const unsigned char tail = static_cast<unsigned char>(text[i + j]);
            if ((tail & 0xC0U) != 0x80U) {
                throw std::runtime_error("Qwen3 forced aligner encountered invalid UTF-8 continuation byte");
            }
            codepoint = (codepoint << 6U) | (tail & 0x3FU);
        }
        out.push_back({codepoint, text.substr(i, len)});
        i += len;
    }
    return out;
}

bool is_cjk(uint32_t codepoint) {
    return (codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF) ||
        (codepoint >= 0x2A700 && codepoint <= 0x2B73F) ||
        (codepoint >= 0x2B740 && codepoint <= 0x2B81F) ||
        (codepoint >= 0x2B820 && codepoint <= 0x2CEAF) ||
        (codepoint >= 0xF900 && codepoint <= 0xFAFF);
}

bool is_ascii_alnum(uint32_t codepoint) {
    return (codepoint >= 'A' && codepoint <= 'Z') ||
        (codepoint >= 'a' && codepoint <= 'z') ||
        (codepoint >= '0' && codepoint <= '9');
}

bool is_kept_non_ascii_letter_or_number(uint32_t codepoint) {
    return (codepoint >= 0x00C0 && codepoint <= 0x02AF) ||
        (codepoint >= 0x0370 && codepoint <= 0x03FF) ||
        (codepoint >= 0x0400 && codepoint <= 0x052F) ||
        (codepoint >= 0x0E00 && codepoint <= 0x0E7F) ||
        (codepoint >= 0x3040 && codepoint <= 0x30FF) ||
        (codepoint >= 0xAC00 && codepoint <= 0xD7AF);
}

bool is_kept_char(uint32_t codepoint) {
    return codepoint == '\'' || is_ascii_alnum(codepoint) || is_cjk(codepoint) ||
        is_kept_non_ascii_letter_or_number(codepoint);
}

std::string clean_token(const std::string & token) {
    std::string out;
    for (const auto & ch : utf8_chars(token)) {
        if (is_kept_char(ch.codepoint)) {
            out += ch.text;
        }
    }
    return out;
}

std::vector<std::string> split_segment_with_chinese(const std::string & segment) {
    std::vector<std::string> out;
    std::string buffer;
    for (const auto & ch : utf8_chars(segment)) {
        if (is_cjk(ch.codepoint)) {
            if (!buffer.empty()) {
                out.push_back(buffer);
                buffer.clear();
            }
            out.push_back(ch.text);
        } else {
            buffer += ch.text;
        }
    }
    if (!buffer.empty()) {
        out.push_back(buffer);
    }
    return out;
}

std::vector<std::string> tokenize_space_language(const std::string & text) {
    std::vector<std::string> words;
    std::string segment;
    auto flush = [&]() {
        if (segment.empty()) {
            return;
        }
        const auto cleaned = clean_token(segment);
        if (!cleaned.empty()) {
            auto split = split_segment_with_chinese(cleaned);
            words.insert(words.end(), split.begin(), split.end());
        }
        segment.clear();
    };
    for (const auto & ch : utf8_chars(text)) {
        if (ch.codepoint == ' ' || ch.codepoint == '\t' || ch.codepoint == '\n' || ch.codepoint == '\r') {
            flush();
        } else {
            segment += ch.text;
        }
    }
    flush();
    return words;
}

std::vector<std::string> tokenize_chinese_mixed(const std::string & text) {
    std::vector<std::string> words;
    std::string latin;
    auto flush_latin = [&]() {
        if (!latin.empty()) {
            const auto cleaned = clean_token(latin);
            if (!cleaned.empty()) {
                words.push_back(cleaned);
            }
            latin.clear();
        }
    };
    for (const auto & ch : utf8_chars(text)) {
        if (is_cjk(ch.codepoint)) {
            flush_latin();
            words.push_back(ch.text);
        } else if (is_kept_char(ch.codepoint)) {
            latin += ch.text;
        } else {
            flush_latin();
        }
    }
    flush_latin();
    return words;
}

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch + ('a' - 'A'));
        }
    }
    return value;
}

}  // namespace

std::vector<std::string> tokenize_alignable_words(
    const std::string & text,
    const std::string & language) {
    const auto normalized_language = lower_ascii(language);
    if (normalized_language == "chinese" || normalized_language == "cantonese") {
        return tokenize_chinese_mixed(text);
    }
    return tokenize_space_language(text);
}

bool has_alignable_words(
    const std::string & text,
    const std::string & language) {
    return !tokenize_alignable_words(text, language).empty();
}

namespace {

std::vector<int64_t> fix_timestamp(const std::vector<int64_t> & data) {
    const size_t n = data.size();
    if (n == 0) {
        return {};
    }
    std::vector<int64_t> dp(n, 1);
    std::vector<int64_t> parent(n, -1);
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = static_cast<int64_t>(j);
            }
        }
    }
    const auto max_it = std::max_element(dp.begin(), dp.end());
    size_t index = static_cast<size_t>(std::distance(dp.begin(), max_it));
    std::vector<bool> normal(n, false);
    while (true) {
        normal[index] = true;
        if (parent[index] < 0) {
            break;
        }
        index = static_cast<size_t>(parent[index]);
    }

    std::vector<int64_t> result = data;
    size_t i = 0;
    while (i < n) {
        if (normal[i]) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && !normal[j]) {
            ++j;
        }
        const size_t anomaly_count = j - i;
        bool has_left = false;
        bool has_right = false;
        int64_t left = 0;
        int64_t right = 0;
        for (size_t k = i; k > 0; --k) {
            if (normal[k - 1]) {
                has_left = true;
                left = result[k - 1];
                break;
            }
        }
        for (size_t k = j; k < n; ++k) {
            if (normal[k]) {
                has_right = true;
                right = result[k];
                break;
            }
        }
        if (anomaly_count <= 2) {
            for (size_t k = i; k < j; ++k) {
                if (!has_left && !has_right) {
                    result[k] = data[k];
                } else if (!has_left) {
                    result[k] = right;
                } else if (!has_right) {
                    result[k] = left;
                } else {
                    result[k] = (k - (i - 1)) <= (j - k) ? left : right;
                }
            }
        } else {
            for (size_t k = i; k < j; ++k) {
                if (has_left && has_right) {
                    const double step = static_cast<double>(right - left) / static_cast<double>(anomaly_count + 1);
                    result[k] = static_cast<int64_t>(left + step * static_cast<double>(k - i + 1));
                } else if (has_left) {
                    result[k] = left;
                } else if (has_right) {
                    result[k] = right;
                }
            }
        }
        i = j;
    }
    return result;
}

}  // namespace

Qwen3ForcedAlignProcessor::Qwen3ForcedAlignProcessor(
    const engine::models::qwen3_asr::Qwen3ASRAssets & assets,
    const engine::models::qwen3_asr::Qwen3ASRTextTokenizer & tokenizer)
    : assets_(assets),
      tokenizer_(tokenizer) {}

ForcedAlignPrompt Qwen3ForcedAlignProcessor::build_prompt(
    const std::string & text,
    const std::string & language,
    int64_t audio_feature_tokens) const {
    auto words = tokenize_alignable_words(text, language);
    if (words.empty()) {
        throw std::runtime_error("Qwen3 forced aligner requires non-empty normalized text");
    }
    std::string aligner_text = "<|audio_start|><|audio_pad|><|audio_end|>";
    for (const auto & word : words) {
        aligner_text += word;
        aligner_text += "<timestamp><timestamp>";
    }
    auto prompt = tokenizer_.build_raw_audio_prompt(aligner_text, audio_feature_tokens);
    std::vector<int32_t> timestamp_positions;
    const auto timestamp_token = static_cast<int32_t>(assets_.config.timestamp_token_id);
    for (size_t i = 0; i < prompt.input_ids.size(); ++i) {
        if (prompt.input_ids[i] == timestamp_token) {
            timestamp_positions.push_back(static_cast<int32_t>(i));
        }
    }
    if (timestamp_positions.size() != words.size() * 2) {
        throw std::runtime_error("Qwen3 forced aligner timestamp placeholder count mismatch");
    }
    return {std::move(prompt), std::move(words), std::move(timestamp_positions)};
}

std::vector<engine::runtime::WordTimestamp> Qwen3ForcedAlignProcessor::parse_timestamps(
    const std::vector<std::string> & words,
    const std::vector<int32_t> & timestamp_ids,
    int sample_rate,
    int64_t audio_frames,
    bool clamp_timestamps_to_audio) const {
    if (timestamp_ids.size() != words.size() * 2) {
        throw std::runtime_error("Qwen3 forced aligner timestamp prediction count mismatch");
    }
    if (assets_.config.timestamp_segment_time_ms <= 0) {
        throw std::runtime_error("Qwen3 forced aligner model is missing timestamp_segment_time");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("Qwen3 forced aligner sample_rate must be positive");
    }
    std::vector<int64_t> timestamp_ms;
    timestamp_ms.reserve(timestamp_ids.size());
    for (const int32_t id : timestamp_ids) {
        if (id < 0) {
            throw std::runtime_error("Qwen3 forced aligner produced negative timestamp id");
        }
        timestamp_ms.push_back(static_cast<int64_t>(id) * assets_.config.timestamp_segment_time_ms);
    }
    timestamp_ms = fix_timestamp(timestamp_ms);
    std::vector<engine::runtime::WordTimestamp> out;
    out.reserve(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        const int64_t start_ms = timestamp_ms[i * 2];
        const int64_t end_ms = timestamp_ms[i * 2 + 1];
        engine::runtime::WordTimestamp timestamp;
        timestamp.span.start_sample =
            static_cast<int64_t>(std::llround(static_cast<double>(start_ms) * sample_rate / 1000.0));
        timestamp.span.end_sample =
            static_cast<int64_t>(std::llround(static_cast<double>(end_ms) * sample_rate / 1000.0));
        timestamp.word = words[i];
        timestamp.confidence = 0.0F;
        out.push_back(std::move(timestamp));
    }
    // The official Qwen3 forced aligner can emit equal start/end timestamps for
    // some transcript tokenizations. Keep the predicted ordering, but repair
    // zero-length spans before exposing framework WordTimestamp output.
    const int64_t min_span_samples = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(
            static_cast<double>(assets_.config.timestamp_segment_time_ms) * sample_rate / 1000.0)));
    int64_t previous_end = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        auto & timestamp = out[i];
        const int64_t remaining_words = static_cast<int64_t>(out.size() - i - 1);
        timestamp.span.start_sample = std::max(timestamp.span.start_sample, previous_end);
        if (clamp_timestamps_to_audio && audio_frames > 0) {
            const int64_t latest_start = std::max<int64_t>(0, audio_frames - remaining_words - 1);
            timestamp.span.start_sample = std::min(timestamp.span.start_sample, latest_start);
        }
        if (timestamp.span.end_sample <= timestamp.span.start_sample) {
            const int64_t min_end = timestamp.span.start_sample + min_span_samples;
            if (i + 1 < out.size() && out[i + 1].span.start_sample > timestamp.span.start_sample) {
                timestamp.span.end_sample = std::min(min_end, out[i + 1].span.start_sample);
            } else {
                timestamp.span.end_sample = min_end;
            }
        }
        if (clamp_timestamps_to_audio && audio_frames > 0) {
            const int64_t latest_end = std::max<int64_t>(1, audio_frames - remaining_words);
            timestamp.span.end_sample = std::min(timestamp.span.end_sample, latest_end);
            if (timestamp.span.end_sample <= timestamp.span.start_sample) {
                timestamp.span.start_sample = std::max<int64_t>(0, timestamp.span.end_sample - 1);
            }
        }
        previous_end = timestamp.span.end_sample;
    }
    return out;
}

}  // namespace engine::models::qwen3_forced_aligner

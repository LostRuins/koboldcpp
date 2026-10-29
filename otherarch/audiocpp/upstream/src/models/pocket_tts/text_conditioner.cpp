#include "engine/models/pocket_tts/text_conditioner.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/pocket_tts/assets.h"
#include "graph_common.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_set>

namespace engine::models::pocket_tts {
namespace {

struct CodePoint {
    uint32_t value = 0;
    size_t length = 0;
};

// Decodes the UTF-8 sequence starting at `start`. A malformed sequence is returned as a
// single opaque byte, so callers never read past it and never treat it as a letter.
CodePoint decode_utf8_at(const std::string & text, size_t start) {
    const auto lead = static_cast<unsigned char>(text[start]);
    if (lead < 0x80) {
        return {lead, 1};
    }
    size_t length = 0;
    uint32_t value = 0;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
        value = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        value = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        value = lead & 0x07;
    } else {
        return {0xFFFFFFFFU, 1};
    }
    if (start + length > text.size()) {
        return {0xFFFFFFFFU, 1};
    }
    for (size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[start + i]);
        if ((next & 0xC0) != 0x80) {
            return {0xFFFFFFFFU, 1};
        }
        value = (value << 6) | (next & 0x3F);
    }
    return {value, length};
}

std::string encode_utf8(uint32_t value) {
    std::string out;
    if (value < 0x80) {
        out.push_back(static_cast<char>(value));
    } else if (value < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else if (value < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (value >> 18)));
        out.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
    return out;
}

// The reference works on Python str, where "é".isalnum() is true. Testing the last BYTE of
// a UTF-8 string instead left every sentence ending in an accented letter without its
// final period ("C'est réglé"). Covers ASCII plus the Latin-1 Supplement and Latin
// Extended-A/B letters used by every Pocket TTS language; other scripts keep the byte rule.
bool is_letter_or_digit(uint32_t value) {
    if (value < 0x80) {
        return std::isalnum(static_cast<int>(value)) != 0;
    }
    return value >= 0x00C0 && value <= 0x024F && value != 0x00D7 && value != 0x00F7;
}

// Uppercase of a lowercase Latin letter, or 0 when it has no single-code-point uppercase:
// the reference capitalizes the first letter with str.upper(), which turns "état" into
// "État" where a byte-wise toupper() left it lowercase.
uint32_t latin_uppercase(uint32_t value) {
    if (value >= 'a' && value <= 'z') {
        return value - 0x20;
    }
    if (value >= 0x00E0 && value <= 0x00FE && value != 0x00F7) {
        return value - 0x20;
    }
    if (value == 0x00FF) {
        return 0x0178;
    }
    if (value == 0x0131) {
        return 'I';
    }
    const bool even_upper = (value >= 0x0100 && value <= 0x012F) || (value >= 0x0132 && value <= 0x0137) ||
        (value >= 0x014A && value <= 0x0177);
    if (even_upper && (value & 1U) == 1U) {
        return value - 1;
    }
    const bool odd_upper = (value >= 0x0139 && value <= 0x0148) || (value >= 0x0179 && value <= 0x017E);
    if (odd_upper && (value & 1U) == 0U) {
        return value - 1;
    }
    return 0;
}

std::string strip_ascii_whitespace(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

std::string prepare_pocket_tts_text(
    std::string text,
    bool pad_with_spaces_for_short_inputs,
    bool remove_semicolons) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    if (text.empty()) {
        throw std::runtime_error("PocketTTS text prompt cannot be empty");
    }

    for (char & ch : text) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    for (size_t i = 1; i < text.size();) {
        if (text[i] == ' ' && text[i - 1] == ' ') {
            text.erase(text.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    if (remove_semicolons) {
        std::replace(text.begin(), text.end(), ';', ',');
    }
    const CodePoint first = decode_utf8_at(text, 0);
    if (const uint32_t upper = latin_uppercase(first.value); upper != 0) {
        text.replace(0, first.length, encode_utf8(upper));
    }
    size_t last_start = text.size() - 1;
    while (last_start > 0 && (static_cast<unsigned char>(text[last_start]) & 0xC0) == 0x80) {
        --last_start;
    }
    const CodePoint last = decode_utf8_at(text, last_start);
    if (last_start + last.length == text.size() && is_letter_or_digit(last.value)) {
        text.push_back('.');
    }
    if (pad_with_spaces_for_short_inputs) {
        int words = 0;
        bool in_word = false;
        for (const char ch : text) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (in_word) {
                    ++words;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if (in_word) {
            ++words;
        }
        if (words < 5) {
            text = "        " + text;
        }
    }
    return text;
}

std::vector<float> gather_token_embeddings(
    const assets::TensorDataF32 & embedding_table,
    const std::vector<int32_t> & tokens,
    int64_t hidden_size) {
    if (embedding_table.shape.rank != 2 || embedding_table.shape.dims[1] != hidden_size) {
        throw std::runtime_error("PocketTTS embedding table must have shape [vocab, hidden_size]");
    }
    const int64_t vocab = embedding_table.shape.dims[0];
    const int64_t dim = embedding_table.shape.dims[1];

    std::vector<float> output(static_cast<size_t>(tokens.size() * dim));
    for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        const int32_t token = tokens[token_index];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("PocketTTS token id is out of range for embedding table");
        }
        const size_t src_offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
        const size_t dst_offset = token_index * static_cast<size_t>(dim);
        std::copy_n(
            embedding_table.values.begin() + static_cast<ptrdiff_t>(src_offset),
            static_cast<size_t>(dim),
            output.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    return output;
}

// Token ids of `symbols` minus the first one, exactly as the reference builds its boundary
// sets: SentencePiece prefixes the string with its word marker, and only the ids that
// follow it stand for the punctuation.
std::unordered_set<int32_t> boundary_token_ids(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::string & symbols) {
    const std::vector<int32_t> ids = tokenizers::tokenize_sentencepiece(pieces, symbols);
    std::unordered_set<int32_t> out;
    for (size_t i = 1; i < ids.size(); ++i) {
        out.insert(ids[i]);
    }
    return out;
}

std::string decode_range(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t begin,
    size_t end) {
    return tokenizers::decode_sentencepiece(
        pieces,
        std::vector<int32_t>(
            tokens.begin() + static_cast<ptrdiff_t>(begin),
            tokens.begin() + static_cast<ptrdiff_t>(end)));
}

// "3.5": a period between two digits does not end a sentence.
bool is_decimal_period_boundary(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t segment_start) {
    const std::string prefix = decode_range(pieces, tokens, 0, segment_start);
    const std::string suffix = decode_range(pieces, tokens, segment_start, tokens.size());
    return prefix.size() >= 2 && prefix.back() == '.' &&
        std::isdigit(static_cast<unsigned char>(prefix[prefix.size() - 2])) != 0 && !suffix.empty() &&
        std::isdigit(static_cast<unsigned char>(suffix.front())) != 0;
}

// A segment starts at the first token that is not a boundary token after a run of boundary
// tokens. Returns [0, starts..., size], so consecutive pairs are the segments.
std::vector<size_t> find_boundary_indices(
    const std::vector<int32_t> & tokens,
    const std::unordered_set<int32_t> & boundary_ids,
    const std::vector<tokenizers::SentencePiecePiece> * pieces_for_decimal_check) {
    std::vector<size_t> indices{0};
    bool previous_was_boundary = false;
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (boundary_ids.count(tokens[index]) != 0) {
            previous_was_boundary = true;
            continue;
        }
        if (previous_was_boundary &&
            !(pieces_for_decimal_check != nullptr &&
              is_decimal_period_boundary(*pieces_for_decimal_check, tokens, index))) {
            indices.push_back(index);
        }
        previous_was_boundary = false;
    }
    indices.push_back(tokens.size());
    return indices;
}

struct TextSegment {
    int64_t token_count = 0;
    std::string text;
};

std::vector<TextSegment> segments_from_boundaries(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    const std::vector<size_t> & boundaries) {
    std::vector<TextSegment> segments;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        segments.push_back(TextSegment{
            static_cast<int64_t>(boundaries[i + 1] - boundaries[i]),
            decode_range(pieces, tokens, boundaries[i], boundaries[i + 1]),
        });
    }
    return segments;
}

}  // namespace

TextConditioner::TextConditioner(TextConditionerConfig config) : config_(config) {}

TextConditioningResult TextConditioner::prepare(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSHostWeights & weights,
    const std::string & text) const {
    TextConditioningResult result;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        result.prepared_text = prepare_pocket_tts_text(
            text,
            manifest.model_config.pad_with_spaces_for_short_inputs,
            manifest.model_config.remove_semicolons);
        if (manifest.tokenizer_pieces.empty()) {
            throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
        }
        result.tokens = tokenizers::tokenize_sentencepiece(manifest.tokenizer_pieces, result.prepared_text);
        const auto & embedding_table = weights.conditioner_embedding_table;
        result.text_embeddings = gather_token_embeddings(embedding_table, result.tokens, config_.hidden_size);
    });
    engine::debug::timing_log_scalar("pocket_tts.text.prepare_ms", prepare_ms);
    return result;
}

std::vector<std::string> TextConditioner::split_into_sentence_chunks(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const std::string & text,
    int64_t max_tokens) const {
    if (manifest.tokenizer_pieces.empty()) {
        throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
    }
    if (max_tokens <= 0) {
        throw std::runtime_error("PocketTTS max_tokens must be positive");
    }
    const auto & pieces = manifest.tokenizer_pieces;
    // Same normalization as the prompt the model will see, then stripped: the reference drops
    // the short-input padding here and adds it back chunk by chunk in prepare().
    const std::string prepared = strip_ascii_whitespace(prepare_pocket_tts_text(
        text,
        manifest.model_config.pad_with_spaces_for_short_inputs,
        manifest.model_config.remove_semicolons));
    const std::vector<int32_t> tokens = tokenizers::tokenize_sentencepiece(pieces, prepared);
    const std::vector<TextSegment> sentences = segments_from_boundaries(
        pieces,
        tokens,
        find_boundary_indices(tokens, boundary_token_ids(pieces, ".!...?"), &pieces));

    // A sentence longer than the budget is split again on commas, semicolons and colons,
    // which keeps the model from skipping words at the end of an overlong chunk.
    const std::unordered_set<int32_t> fallback_ids = boundary_token_ids(pieces, ",;:");
    std::vector<TextSegment> refined;
    for (const TextSegment & sentence : sentences) {
        if (sentence.token_count <= max_tokens) {
            refined.push_back(sentence);
            continue;
        }
        const std::vector<int32_t> sub_tokens =
            tokenizers::tokenize_sentencepiece(pieces, strip_ascii_whitespace(sentence.text));
        std::vector<TextSegment> parts = segments_from_boundaries(
            pieces,
            sub_tokens,
            find_boundary_indices(sub_tokens, fallback_ids, nullptr));
        if (parts.size() > 1) {
            refined.insert(refined.end(), parts.begin(), parts.end());
        } else {
            refined.push_back(sentence);
        }
    }

    std::vector<std::string> chunks;
    std::string current;
    int64_t current_tokens = 0;
    for (const TextSegment & segment : refined) {
        if (current.empty()) {
            current = segment.text;
            current_tokens = segment.token_count;
            continue;
        }
        if (current_tokens + segment.token_count > max_tokens) {
            chunks.push_back(strip_ascii_whitespace(current));
            current = segment.text;
            current_tokens = segment.token_count;
        } else {
            current += " " + segment.text;
            current_tokens += segment.token_count;
        }
    }
    if (!current.empty()) {
        chunks.push_back(strip_ascii_whitespace(current));
    }
    return chunks;
}

}  // namespace engine::models::pocket_tts

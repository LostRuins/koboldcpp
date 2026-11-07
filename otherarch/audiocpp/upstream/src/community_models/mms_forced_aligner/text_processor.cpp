#include "engine/community_models/mms_forced_aligner/text_processor.h"

#include "engine/framework/text/unicode_normalization.h"

#include "unicode.h"

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::mms_forced_aligner {

namespace {

constexpr int32_t kStarId = MmsVocabulary::kStarId;
constexpr int32_t kNoWord = -1;

bool is_keep_char(uint32_t codepoint) {
    return (codepoint >= static_cast<uint32_t>('a') && codepoint <= static_cast<uint32_t>('z')) ||
           codepoint == static_cast<uint32_t>('\'');
}

std::string normalize_latin_word(const std::string & word) {
    const auto codepoints = unicode_cpts_from_utf8(word);
    const auto decomposed = engine::text::normalize_nfkd_codepoints(codepoints);
    std::string out;
    for (const uint32_t codepoint : decomposed) {
        const uint32_t lower = unicode_tolower(codepoint);
        if (is_keep_char(lower)) {
            out.push_back(static_cast<char>(lower));
            continue;
        }
        if (unicode_cpt_flags_from_cpt(codepoint).is_letter) {
            throw std::runtime_error(
                "MMS forced aligner latin normalization does not support the non-Latin letter in '" +
                word + "'; use text_normalization=pre_romanized or romanize the transcript first");
        }
        // Punctuation, digits, combining marks, and whitespace act as separators.
    }
    return out;
}

std::string normalize_pre_romanized_word(const std::string & word) {
    const auto codepoints = unicode_cpts_from_utf8(word);
    std::string out;
    for (const uint32_t codepoint : codepoints) {
        const uint32_t lower = unicode_tolower(codepoint);
        if (is_keep_char(lower)) {
            out.push_back(static_cast<char>(lower));
            continue;
        }
        if (unicode_cpt_flags_from_cpt(codepoint).is_letter) {
            throw std::runtime_error(
                "MMS forced aligner pre_romanized input must be ASCII romanized text; '" + word +
                "' contains a non-ASCII letter");
        }
    }
    return out;
}

int32_t target_id(const MmsVocabulary & vocab, char letter) {
    const std::string token(1, letter);
    const auto it = vocab.token_to_id.find(token);
    if (it == vocab.token_to_id.end()) {
        throw std::runtime_error("MMS forced aligner vocab is missing the '" + token + "' target");
    }
    return it->second;
}

std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> words;
    std::string current;
    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        words.push_back(std::move(current));
    }
    return words;
}

}  // namespace

std::string mms_canonical_language(const std::string & language) {
    if (language == "nl" || language == "nld") {
        return "nld";
    }
    if (language == "en" || language == "eng") {
        return "eng";
    }
    throw std::runtime_error(
        "MMS forced aligner supports nl/nld and en/eng transcripts, got '" + language +
        "'; other languages require pre-romanized text");
}

MmsPreparedText prepare_mms_text(
    const MmsVocabulary & vocab,
    const std::string & text,
    const std::string & language,
    const MmsTextProcessorOptions & options) {
    MmsPreparedText prepared;
    if (options.normalization == MmsTextNormalization::Latin) {
        prepared.canonical_language = mms_canonical_language(language);
    } else {
        if (language.empty()) {
            throw std::runtime_error("MMS forced aligner pre_romanized input requires a language code");
        }
        // Pre-romanized targets are language-agnostic; keep the caller's code
        // for metadata, canonicalizing only the natively supported pair.
        try {
            prepared.canonical_language = mms_canonical_language(language);
        } catch (const std::runtime_error &) {
            prepared.canonical_language = language;
        }
    }

    const auto normalize_word = options.normalization == MmsTextNormalization::Latin
        ? normalize_latin_word
        : normalize_pre_romanized_word;

    for (const auto & original : split_words(text)) {
        const std::string normalized = normalize_word(original);
        if (normalized.empty()) {
            // Standalone numbers and punctuation-only tokens carry no targets.
            continue;
        }
        prepared.original_words.push_back(original);
        prepared.normalized_words.push_back(normalized);
    }
    if (prepared.original_words.empty()) {
        throw std::runtime_error("MMS forced aligner transcript contains no alignable words");
    }

    const auto append_word_targets = [&](size_t word_index) {
        for (const char letter : prepared.normalized_words[word_index]) {
            prepared.target_ids.push_back(target_id(vocab, letter));
            prepared.target_to_word.push_back(static_cast<int32_t>(word_index));
        }
    };
    const auto append_star = [&]() {
        prepared.target_ids.push_back(kStarId);
        prepared.target_to_word.push_back(kNoWord);
    };

    if (options.star_frequency == MmsStarFrequency::Segment) {
        for (size_t index = 0; index < prepared.original_words.size(); ++index) {
            append_star();
            append_word_targets(index);
        }
    } else {
        append_star();
        for (size_t index = 0; index < prepared.original_words.size(); ++index) {
            append_word_targets(index);
        }
        append_star();
    }
    return prepared;
}

}  // namespace engine::community_models::mms_forced_aligner

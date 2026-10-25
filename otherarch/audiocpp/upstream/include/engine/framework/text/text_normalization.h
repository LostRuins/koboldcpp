#pragma once

#include <string>
#include <string_view>

namespace engine::text {

struct EnglishTextNormalizationOptions {
    bool expand_common_contractions = false;
    bool spell_numbers = true;
    bool index_tts_punctuation = false;
    bool uppercase_ascii = false;
    // Verbalize standalone ASCII symbols like the official wetext English
    // normalizer ("a_b" -> "a underscore b", "C++" -> "C plus plus").
    bool verbalize_symbols = false;
};

std::string replace_all(std::string text, std::string_view from, std::string_view to);
std::string collapse_ascii_whitespace(std::string_view text);

std::string normalize_english_numbers(std::string text);

// Applies the IndexTTS punctuation replacement map (、→",", 。→"." etc.),
// matching the char_rep_map the official front.TextNormalizer applies to
// every language path.
std::string normalize_index_tts_punctuation(std::string text);
std::string normalize_english_text(
    std::string_view text,
    const EnglishTextNormalizationOptions & options = {});

// Warning: this is only a lightweight kana/width/punctuation cleanup helper.
// It does not perform Japanese word segmentation or Kanji-to-reading conversion,
// so it is not ready for model frontends that require full Japanese normalization.
std::string normalize_japanese_text(std::string_view text);

}  // namespace engine::text

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace engine::text {

// Returns true if the language tag indicates Cantonese/Yue. Covers
// "yue", "yue-Hant", "zh-yue", "cantonese", "zh-HK", "zh-MO" variants.
// Comparison is case-insensitive and accepts '-' / '_' separators.
bool is_cantonese_language(std::string_view language) noexcept;

// Returns true if the UTF-8 text contains at least one Traditional
// character that would be converted by convert_traditional_to_simplified().
bool contains_traditional_characters(std::string_view text);

// Convert Traditional Chinese characters to Simplified Chinese.
// Uses OpenCC TSCharacters.txt mapping (3222 single-char entries).
// Characters without a mapping are left unchanged. Invalid UTF-8
// throws std::runtime_error (label: "chinese variant text").
std::string convert_traditional_to_simplified(std::string_view text);

// If `language` is Cantonese/Yue, returns text unchanged.
// Otherwise converts Traditional -> Simplified.
// Empty / "auto" / "zh" etc. are treated as non-Cantonese (convert).
std::string maybe_convert_traditional_to_simplified(
    std::string_view text,
    std::string_view language);

// Helper for optional language: std::nullopt / empty means convert.
inline std::string maybe_convert_traditional_to_simplified_opt(
    std::string_view text,
    const std::optional<std::string> & language) {
    if (!language.has_value() || language->empty()) {
        return convert_traditional_to_simplified(text);
    }
    return maybe_convert_traditional_to_simplified(text, std::string_view(*language));
}

}  // namespace engine::text

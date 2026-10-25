#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::text {

inline bool is_utf8_continuation(unsigned char ch) noexcept {
    return (ch & 0xC0U) == 0x80U;
}

inline size_t utf8_codepoint_count(std::string_view text, std::string_view label) {
    size_t count = 0;
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
        pos += width;
        ++count;
    }
    return count;
}

}  // namespace engine::text

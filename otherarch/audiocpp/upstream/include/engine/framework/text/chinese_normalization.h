#pragma once

#include <string>
#include <string_view>

namespace engine::text {

enum class ChineseTextNormalizationTarget {
    IndexTTS,
    Confucius4TTS,
};

class ChineseTextNormalizer {
public:
    explicit ChineseTextNormalizer(ChineseTextNormalizationTarget target = ChineseTextNormalizationTarget::IndexTTS);

    std::string normalize(std::string_view text) const;

private:
    ChineseTextNormalizationTarget target_;
};

std::string normalize_chinese_text(
    std::string_view text,
    ChineseTextNormalizationTarget target = ChineseTextNormalizationTarget::IndexTTS);

}  // namespace engine::text

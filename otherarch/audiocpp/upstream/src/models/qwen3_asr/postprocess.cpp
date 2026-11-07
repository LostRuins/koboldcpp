#include "engine/models/qwen3_asr/postprocess.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace engine::models::qwen3_asr {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

bool starts_with_language(const std::string & value) {
    constexpr const char * kPrefix = "language ";
    if (value.size() < std::char_traits<char>::length(kPrefix)) {
        return false;
    }
    for (size_t i = 0; kPrefix[i] != '\0'; ++i) {
        if (static_cast<char>(std::tolower(static_cast<unsigned char>(value[i]))) != kPrefix[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

Qwen3ASRPostprocessor::Qwen3ASRPostprocessor(const Qwen3ASRTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

Qwen3ASRResult Qwen3ASRPostprocessor::decode(
    const Qwen3ASRGeneratedTokens & tokens,
    const Qwen3ASRRequest & request) const {
    Qwen3ASRResult result;
    const std::string raw = trim(tokenizer_.decode(tokens.token_ids));
    result.language = request.language;
    if (!request.language.empty()) {
        result.text = raw;
        return result;
    }
    constexpr const char * kTag = "<asr_text>";
    const size_t tag = raw.find(kTag);
    if (tag == std::string::npos) {
        result.text = raw;
        return result;
    }
    const std::string meta = trim(raw.substr(0, tag));
    result.text = trim(raw.substr(tag + std::char_traits<char>::length(kTag)));
    if (starts_with_language(meta)) {
        result.language = trim(meta.substr(std::char_traits<char>::length("language ")));
        if (result.language == "None") {
            result.language.clear();
            if (result.text.empty()) {
                return result;
            }
        }
    }
    return result;
}

}  // namespace engine::models::qwen3_asr

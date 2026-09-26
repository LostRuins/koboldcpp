#include "tokenize_util.h"

std::vector<std::string> split_with_special_tokens(
    const std::string& text,
    const std::vector<std::string>& special_tokens) {
    std::vector<std::string> result;
    size_t pos      = 0;
    size_t text_len = text.size();

    while (pos < text_len) {
        size_t next_pos = text_len;
        std::string matched_token;

        for (const auto& token : special_tokens) {
            if (token.empty()) {
                continue;
            }
            size_t token_pos = text.find(token, pos);
            if (token_pos != std::string::npos &&
                (token_pos < next_pos || (token_pos == next_pos && token.size() > matched_token.size()))) {
                next_pos      = token_pos;
                matched_token = token;
            }
        }

        if (next_pos > pos) {
            result.push_back(text.substr(pos, next_pos - pos));
        }

        if (!matched_token.empty()) {
            result.push_back(matched_token);
            pos = next_pos + matched_token.size();
        } else {
            break;
        }
    }

    return result;
}

#include "engine/framework/io/text.h"

#include <algorithm>
#include <cctype>

namespace engine::io {

std::string trim_ascii_whitespace(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(
        value.begin(),
        std::find_if(
            value.begin(),
            value.end(),
            [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); })
            .base(),
        value.end());
    return value;
}

}  // namespace engine::io

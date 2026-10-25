#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::text {

enum class TextChunkMode {
    Default,
    TagAware,
    Japanese,
    Endline,
};

std::string_view text_chunk_mode_name(TextChunkMode mode);

std::optional<int64_t> parse_text_chunk_size_override(
    const std::unordered_map<std::string, std::string> & options);

std::optional<TextChunkMode> parse_text_chunk_mode_override(
    const std::unordered_map<std::string, std::string> & options);

std::vector<std::string> split_text_chunks(
    std::string_view text,
    int64_t codepoint_budget);

std::vector<std::string> split_text_chunks(
    std::string_view text,
    int64_t codepoint_budget,
    TextChunkMode mode);

}  // namespace engine::text

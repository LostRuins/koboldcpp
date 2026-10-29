#pragma once

#include <cstdint>
#include <vector>

namespace engine::text {

bool append_known_unicode_decomposition(uint32_t codepoint, std::vector<uint32_t> & out);

std::vector<uint32_t> decompose_known_unicode_codepoints(const std::vector<uint32_t> & codepoints);

bool append_nfkd_decomposition(uint32_t codepoint, std::vector<uint32_t> & out);

std::vector<uint32_t> normalize_nfkd_codepoints(const std::vector<uint32_t> & codepoints);

}  // namespace engine::text

#ifndef __SD_TOKENIZERS_BPE_TOKENIZE_UTIL_H__
#define __SD_TOKENIZERS_BPE_TOKENIZE_UTIL_H__

#include <string>
#include <vector>

std::vector<std::string> token_split(const std::string& text);
std::vector<std::string> split_with_special_tokens(const std::string& text, const std::vector<std::string>& special_tokens);

#endif  // __SD_TOKENIZERS_BPE_TOKENIZE_UTIL_H__
#ifndef __TOKENIZE_UTIL__
#define __TOKENIZE_UTIL__

#include <string>
#include <vector>

std::vector<std::string> token_split(const std::string& text);
std::vector<std::string> split_with_special_tokens(const std::string& text, const std::vector<std::string>& special_tokens);

#endif  // __TOKENIZE_UTIL__
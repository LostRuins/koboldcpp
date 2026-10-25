#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {

struct TokenizedText {
    std::string text;
    std::vector<int32_t> token_ids;
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::string family() const = 0;
    virtual TokenizedText tokenize(const std::string & text) const = 0;
};

using TokenizerPtr = std::shared_ptr<ITokenizer>;

}  // namespace engine::tokenizers

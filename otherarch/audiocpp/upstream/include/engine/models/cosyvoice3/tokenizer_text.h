#pragma once

#include "engine/models/cosyvoice3/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::cosyvoice3 {

struct CosyVoice3TextTokens {
    std::vector<int32_t> prompt;
    std::vector<int32_t> target;
};

class CosyVoice3TextTokenizer {
public:
    explicit CosyVoice3TextTokenizer(std::shared_ptr<const CosyVoice3Assets> assets);
    ~CosyVoice3TextTokenizer();

    CosyVoice3TextTokens encode_zero_shot(std::string_view text, std::string_view prompt_text) const;
    CosyVoice3TextTokens encode_cross_lingual(std::string_view text) const;
    CosyVoice3TextTokens encode_instruct(std::string_view text, std::string_view instruction) const;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::cosyvoice3

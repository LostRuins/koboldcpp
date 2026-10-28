#pragma once

#include "engine/models/omnivoice/assets.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

class OmniVoiceTextTokenizer {
public:
    struct Impl;

    explicit OmniVoiceTextTokenizer(std::shared_ptr<const OmniVoiceAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<int32_t> encode_with_nonverbal_tags(const std::string & text) const;
    int32_t token_id(const std::string & token) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::omnivoice

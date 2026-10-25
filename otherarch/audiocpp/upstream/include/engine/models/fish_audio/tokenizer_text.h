#pragma once

#include "engine/models/fish_audio/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::fish_audio {

class FishAudioTextTokenizer {
public:
    struct Impl;

    explicit FishAudioTextTokenizer(std::shared_ptr<const FishAudioAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    int32_t token_id(const std::string & token) const;
    int32_t im_end_id() const noexcept;
    int32_t semantic_begin_id() const noexcept;
    int32_t semantic_end_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::fish_audio

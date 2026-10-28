#pragma once

#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxcpm2 {

class VoxCPM2TextTokenizer {
public:
    struct Impl;

    explicit VoxCPM2TextTokenizer(std::shared_ptr<const VoxCPM2Assets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    VoxCPM2TextPrompt build_prompt(const std::string & text) const;
    int32_t audio_start_token_id() const noexcept;
    int32_t audio_end_token_id() const noexcept;
    int32_t reference_audio_start_token_id() const noexcept;
    int32_t reference_audio_end_token_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::voxcpm2


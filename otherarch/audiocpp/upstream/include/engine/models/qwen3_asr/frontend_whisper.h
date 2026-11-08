#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/types.h"

#include <memory>

namespace engine::models::qwen3_asr {

class Qwen3ASRWhisperFrontend {
public:
    explicit Qwen3ASRWhisperFrontend(std::shared_ptr<const Qwen3ASRAssets> assets);

    Qwen3ASRAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const Qwen3ASRAssets> assets_;
    engine::audio::WhisperLogMelExtractor extractor_;
};

}  // namespace engine::models::qwen3_asr

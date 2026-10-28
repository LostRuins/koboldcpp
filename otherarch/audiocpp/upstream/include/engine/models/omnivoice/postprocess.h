#pragma once

#include "engine/models/omnivoice/types.h"

namespace engine::models::omnivoice {

class OmniVoicePostprocessor {
public:
    OmniVoiceResult finalize(const runtime::AudioBuffer & audio, const OmniVoiceRequest & request) const;
};

}  // namespace engine::models::omnivoice

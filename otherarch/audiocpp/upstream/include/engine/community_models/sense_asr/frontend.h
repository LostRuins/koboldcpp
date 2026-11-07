#pragma once

#include "engine/community_models/sense_asr/assets.h"
#include "engine/community_models/sense_asr/types.h"

#include <vector>

namespace engine::community_models::sense_asr {

class SenseAsrFrontend {
public:
  explicit SenseAsrFrontend(SenseAsrFrontendConfig config);

  SenseAsrAudioFeatures extract(const std::vector<float> &audio,
                                int sample_rate) const;

private:
  SenseAsrFrontendConfig config_;
};

} // namespace engine::community_models::sense_asr

#pragma once

#include <cstdint>
#include <vector>

namespace engine::decoders {

struct TdtDecodeResult {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> token_timestamps;
    std::vector<int32_t> token_durations;
};

struct TdtJointStep {
    int32_t label = 0;
    float label_score = 0.0f;
    int32_t duration_index = 0;
};

struct TdtPredictorStateSnapshot {
    std::vector<float> values;
};

}  // namespace engine::decoders

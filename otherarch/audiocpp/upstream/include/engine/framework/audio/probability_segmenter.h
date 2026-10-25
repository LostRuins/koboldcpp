#pragma once

#include <cstdint>
#include <optional>

namespace engine::audio {

struct ProbabilitySegment {
    int64_t start = 0;
    int64_t end = 0;
    float confidence = 0.0f;
};

struct ProbabilitySegmenterConfig {
    float threshold = 0.5f;
    int64_t min_speech = 0;
    int64_t min_silence = 0;
};

// Positions use caller-supplied units. Overlapping windows end at the last
// positive window's end, not at the first negative window's start.
class ProbabilitySegmenter {
public:
    explicit ProbabilitySegmenter(ProbabilitySegmenterConfig config);
    std::optional<ProbabilitySegment> push(float probability, int64_t start, int64_t end);
    std::optional<ProbabilitySegment> finish();

private:
    ProbabilitySegmenterConfig config_;
    bool active_ = false;
    int64_t start_ = 0;
    int64_t end_ = 0;
    double confidence_sum_ = 0.0;
    int64_t confidence_count_ = 0;
};

}  // namespace engine::audio

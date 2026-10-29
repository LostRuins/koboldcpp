#pragma once

#include <cstdint>
#include <vector>

namespace engine::audio {

struct AudioActivityRegion {
    bool has_activity = false;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
};

AudioActivityRegion find_interleaved_audio_activity_region(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    int sample_rate_hz,
    int64_t start_frame,
    int64_t frame_count,
    float threshold_dbfs,
    double window_seconds,
    double margin_seconds);

}  // namespace engine::audio

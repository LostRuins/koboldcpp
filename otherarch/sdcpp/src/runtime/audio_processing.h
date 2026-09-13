#ifndef __SD_RUNTIME_AUDIO_PROCESSING_H__
#define __SD_RUNTIME_AUDIO_PROCESSING_H__

#include <cstdint>
#include <vector>

namespace sd::audio {

    // Returns the input unchanged when sample rates are equal, and an empty vector on invalid input.
    std::vector<float> resample_audio(const float* samples,
                                      uint64_t sample_count,
                                      uint32_t orig_sample_rate,
                                      uint32_t target_sample_rate);

    // Average interleaved channels; return an empty vector on invalid input.
    std::vector<float> downmix_to_mono(const float* interleaved_samples,
                                       uint64_t sample_count,
                                       uint32_t channels);

}  // namespace sd::audio

#endif  // __SD_RUNTIME_AUDIO_PROCESSING_H__

#ifndef __SD_CONDITIONING_WAN_AUDIO_H__
#define __SD_CONDITIONING_WAN_AUDIO_H__

#include <vector>

namespace sd::wan_audio {

    struct BucketPlan {
        int audio_frames;  // frames at video_rate
        int batch_frames;  // latent_t * 4
        int video_rate;
        int fps;         // bucket frame rate
        int num_chunks;  // includes trailing padding
        int bucket_frames;
        int padded_audio_frames;
    };

    // [layers, frames, dim] at input_fps -> [bucket_frames, layers, dim] at fps.
    // Pads past the audio end; returns an empty vector on invalid input.
    std::vector<float> build_audio_buckets(const float* stacked_states,
                                           int num_layers,
                                           int in_frames,
                                           int dim,
                                           int batch_frames,
                                           BucketPlan* plan_out = nullptr,
                                           int input_fps        = 50,
                                           int video_rate       = 30,
                                           int fps              = 16);

}  // namespace sd::wan_audio

#endif  // __SD_CONDITIONING_WAN_AUDIO_H__

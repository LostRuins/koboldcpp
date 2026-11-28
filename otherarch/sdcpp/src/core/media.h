#ifndef __SD_CORE_MEDIA_H__
#define __SD_CORE_MEDIA_H__

#include <cstdint>

namespace sd {
    struct Image {
        uint32_t width;
        uint32_t height;
        uint32_t channel;
        uint8_t* data;
    };

    struct Audio {
        uint32_t sample_rate;
        uint32_t channels;
        uint64_t sample_count;
        float* data;
    };

    struct ReferenceVideo {
        Image* frames;
        int frame_count;
        int fps;
        Audio audio;
    };

    bool preprocess_canny(Image image, float high_threshold, float low_threshold, float weak, float strong, bool inverse);
    void free_image(Image* image);
    void free_audio(Audio* audio);
}

#endif  // __SD_CORE_MEDIA_H__

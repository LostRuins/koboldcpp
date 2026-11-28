#ifndef SD_RESULT_H
#define SD_RESULT_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "tensor.h"

#include "json.hpp"
#include "stable-diffusion.h"

namespace sd {
    struct ImageResult {
        sd::Image image;
        // Owned UTF-8 JSON using the sdcpp.generation/v1 schema.
        char* generation_info;
    };

    struct ImageGenerationResult {
        sd::ImageResult* items;
        int count;
    };

    struct VideoGenerationResult {
        sd::Image* frames;
        int frame_count;
        sd::Audio* audio;
        int fps;
        char* generation_info;
    };

    void free_image_generation_result(ImageGenerationResult* result);
    void free_video_generation_result(VideoGenerationResult* result);
    void free_images(sd::Image* images, int count);

    char* copy_generation_info(const nlohmann::json& info);
    // Consumes images even on failure.
    bool make_image_generation_result(sd::Image* images, int count, const std::vector<nlohmann::json>& infos, sd::ImageGenerationResult** result_out);
    bool make_image_result(ImageGenerationResult* images, sd_result_t** result_out);
    bool make_image_result(sd::Image* images, int count, const std::vector<nlohmann::json>& infos, sd_result_t** result_out);
    bool make_video_result(VideoGenerationResult* video, sd_result_t** result_out);
}

struct sd_result {
    sd_result_kind_t kind = SD_RESULT_INVALID;
    std::vector<sd::api::TensorRef> images;
    std::vector<std::string> generation_infos;
    sd::api::TensorRef audio_tensor;
    sd_audio_t audio{};
    int fps = 0;
    mutable std::mutex video_mutex;
    mutable sd::api::TensorRef video_tensor;
    mutable sd_video_t video{};
};

#endif  // SD_RESULT_H

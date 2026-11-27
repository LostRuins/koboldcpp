#include "result.h"
#include "api.h"

#include <cstdlib>
#include <cstring>
#include <utility>

namespace sd {
    using nlohmann::json;

    char* copy_generation_info(const json& info) {
        const std::string text = info.dump(-1, ' ', false, json::error_handler_t::replace);
        char* result           = static_cast<char*>(malloc(text.size() + 1));
        if (result)
            memcpy(result, text.c_str(), text.size() + 1);
        return result;
    }

    bool make_image_generation_result(sd::Image* images, int count, const std::vector<json>& infos, sd::ImageGenerationResult** result_out) {
        if (!result_out || !images || count <= 0 || infos.size() != static_cast<size_t>(count)) {
            sd::free_images(images, count);
            return false;
        }
        auto* result = static_cast<sd::ImageGenerationResult*>(calloc(1, sizeof(sd::ImageGenerationResult)));
        if (!result) {
            sd::free_images(images, count);
            return false;
        }
        result->items = static_cast<sd::ImageResult*>(calloc(count, sizeof(sd::ImageResult)));
        if (!result->items) {
            free(result);
            sd::free_images(images, count);
            return false;
        }
        result->count = count;
        try {
            for (int i = 0; i < count; ++i) {
                result->items[i].image           = images[i];
                images[i]                        = {};
                result->items[i].generation_info = copy_generation_info(infos[i]);
                if (!result->items[i].generation_info) {
                    sd::free_images(images, count);
                    sd::free_image_generation_result(result);
                    return false;
                }
            }
        } catch (...) {
            sd::free_images(images, count);
            sd::free_image_generation_result(result);
            return false;
        }
        free(images);
        *result_out = result;
        return true;
    }
}

void sd::free_image_generation_result(sd::ImageGenerationResult* result) {
    if (!result)
        return;
    for (int i = 0; i < result->count; ++i) {
        free(result->items[i].image.data);
        free(result->items[i].generation_info);
    }
    free(result->items);
    free(result);
}

void sd::free_video_generation_result(sd::VideoGenerationResult* result) {
    if (!result)
        return;
    sd::free_images(result->frames, result->frame_count);
    sd::free_audio(result->audio);
    free(result->generation_info);
    free(result);
}

static sd::api::TensorRef consume_image(sd::Image& image) {
    std::vector<int64_t> shape{image.channel, image.width, image.height};
    auto* data = image.data;
    image.data = nullptr;
    return sd::api::own_tensor(SD_TENSOR_U8, std::move(shape), data);
}

bool sd::make_image_result(ImageGenerationResult* images, sd_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    std::unique_ptr<ImageGenerationResult, decltype(&free_image_generation_result)> owner(images, free_image_generation_result);
    if (!result_out || !images)
        return false;
    auto result  = std::make_unique<sd_result_t>();
    result->kind = SD_RESULT_IMAGE;
    result->images.reserve(images->count);
    result->generation_infos.reserve(images->count);
    for (int i = 0; i < images->count; ++i) {
        result->images.push_back(consume_image(images->items[i].image));
        result->generation_infos.emplace_back(images->items[i].generation_info ? images->items[i].generation_info : "");
    }
    *result_out = result.release();
    return true;
}

bool sd::make_image_result(sd::Image* images, int count, const std::vector<nlohmann::json>& infos, sd_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    ImageGenerationResult* raw = nullptr;
    if (!make_image_generation_result(images, count, infos, &raw))
        return false;
    return make_image_result(raw, result_out);
}

bool sd::make_video_result(VideoGenerationResult* video, sd_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    std::unique_ptr<VideoGenerationResult, decltype(&free_video_generation_result)> owner(video, free_video_generation_result);
    if (!result_out || !video)
        return false;
    auto result  = std::make_unique<sd_result_t>();
    result->kind = SD_RESULT_VIDEO;
    result->fps  = video->fps;
    result->images.reserve(video->frame_count);
    for (int i = 0; i < video->frame_count; ++i)
        result->images.push_back(consume_image(video->frames[i]));
    result->generation_infos.emplace_back(video->generation_info ? video->generation_info : "");
    if (video->audio) {
        auto& audio = *video->audio;
        std::vector<int64_t> shape{audio.channels, static_cast<int64_t>(audio.sample_count)};
        auto* data           = audio.data;
        audio.data           = nullptr;
        result->audio_tensor = sd::api::own_tensor(SD_TENSOR_F32, std::move(shape), data);
        result->audio        = {audio.sample_rate, result->audio_tensor.get()};
    }
    *result_out = result.release();
    return true;
}

void sd::free_images(sd::Image* result_images, int num_images) {
    if (result_images == nullptr) {
        return;
    }

    for (int i = 0; i < num_images; ++i) {
        if (result_images[i].data != nullptr) {
            free(result_images[i].data);
            result_images[i].data = nullptr;
        }
    }

    free(result_images);
}

void sd::free_audio(sd::Audio* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

void sd::free_image(sd::Image* image) {
    if (image) {
        free(image->data);
        *image = {};
    }
}

void sd_result_free(sd_result_t* result) {
    delete result;
}

sd_result_kind_t sd_result_get_kind(const sd_result_t* result) {
    sd::api::clear_error();
    if (!result) {
        sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result is required");
        return SD_RESULT_INVALID;
    }
    return result->kind;
}

size_t sd_result_get_image_count(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return 0;
    return result->images.size();
}

const sd_tensor_t* sd_result_get_image(const sd_result_t* result, size_t index) {
    if (index >= sd_result_get_image_count(result)) {
        sd::api::error(result ? SD_STATUS_OUT_OF_RANGE : SD_STATUS_INVALID_ARGUMENT, nullptr, "invalid result or image index");
        return nullptr;
    }
    return result->images[index].get();
}

const char* sd_result_get_generation_info(const sd_result_t* result, size_t index) {
    if (!sd_result_get_image(result, index))
        return nullptr;
    return result->generation_infos[result->kind == SD_RESULT_VIDEO ? 0 : index].c_str();
}

const sd_audio_t* sd_result_get_audio(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return nullptr;
    return result->audio.data ? &result->audio : nullptr;
}

int sd_result_get_fps(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return 0;
    return result->fps;
}

const sd_video_t* sd_result_get_video(const sd_result_t* result) {
    const sd_video_t* output = nullptr;
    sd::api::call([&] {
        if (!result)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result is required");
        if (result->kind != SD_RESULT_VIDEO)
            return sd::api::error(SD_STATUS_TYPE_MISMATCH, nullptr, "result is not a video");
        std::lock_guard<std::mutex> lock(result->video_mutex);
        if (!result->video_tensor) {
            if (result->images.empty())
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "video has no frames");
            const auto& first = *result->images.front().get();
            auto shape        = first.shape;
            shape.push_back(static_cast<int64_t>(result->images.size()));
            for (const auto& frame : result->images)
                if (frame->type != first.type || frame->shape != first.shape)
                    return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "video frames must have identical shapes and types");
            auto packed       = sd::api::allocate_tensor(first.type, std::move(shape));
            auto* destination = static_cast<uint8_t*>(packed.get()->mutable_data());
            for (const auto& frame : result->images) {
                memcpy(destination, frame->data(), frame->bytes);
                destination += frame->bytes;
            }
            result->video_tensor = std::move(packed);
            result->video        = {result->video_tensor.get(), result->fps, result->audio};
        }
        output = &result->video;
        return SD_STATUS_OK;
    });
    return output;
}

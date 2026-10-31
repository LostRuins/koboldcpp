#ifndef __EXAMPLE_RESOURCE_OWNERS_H__
#define __EXAMPLE_RESOURCE_OWNERS_H__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "generation_metadata.h"
#include "stable-diffusion.h"

struct FreeDeleter {
    void operator()(void* ptr) const {
        free(ptr);
    }
};

struct FileCloser {
    void operator()(FILE* file) const {
        if (file != nullptr) {
            fclose(file);
        }
    }
};

struct SDCtxDeleter {
    void operator()(sd_ctx_t* ctx) const {
        if (ctx != nullptr) {
            free_sd_ctx(ctx);
        }
    }
};

struct UpscalerCtxDeleter {
    void operator()(upscaler_ctx_t* ctx) const {
        if (ctx != nullptr) {
            free_upscaler_ctx(ctx);
        }
    }
};

struct ADetailerCtxDeleter {
    void operator()(adetailer_ctx_t* ctx) const {
        if (ctx != nullptr) {
            free_adetailer_ctx(ctx);
        }
    }
};

template <typename T>
using FreeUniquePtr = std::unique_ptr<T, FreeDeleter>;

using FilePtr         = std::unique_ptr<FILE, FileCloser>;
using SDCtxPtr        = std::unique_ptr<sd_ctx_t, SDCtxDeleter>;
using UpscalerCtxPtr  = std::unique_ptr<upscaler_ctx_t, UpscalerCtxDeleter>;
using ADetailerCtxPtr = std::unique_ptr<adetailer_ctx_t, ADetailerCtxDeleter>;

class SDImageOwner {
private:
    static sd_image_t copy_image(const sd_image_t& image) {
        if (image.data == nullptr) {
            return {image.width, image.height, image.channel, nullptr};
        }

        const size_t byte_count = static_cast<size_t>(image.width) * image.height * image.channel;
        uint8_t* raw_copy       = static_cast<uint8_t*>(malloc(byte_count));
        if (raw_copy == nullptr) {
            return {0, 0, 0, nullptr};
        }

        std::memcpy(raw_copy, image.data, byte_count);
        return {image.width, image.height, image.channel, raw_copy};
    }

    sd_image_t image_ = {0, 0, 0, nullptr};

public:
    SDImageOwner() = default;
    explicit SDImageOwner(sd_image_t image)
        : image_(image) {
    }

    SDImageOwner(const SDImageOwner& other)
        : image_(copy_image(other.image_)) {
    }

    SDImageOwner& operator=(const SDImageOwner& other) {
        if (this != &other) {
            reset(copy_image(other.image_));
        }
        return *this;
    }

    SDImageOwner(SDImageOwner&& other) noexcept
        : image_(other.release()) {
    }

    SDImageOwner& operator=(SDImageOwner&& other) noexcept {
        if (this != &other) {
            reset();
            image_ = other.release();
        }
        return *this;
    }

    ~SDImageOwner() {
        reset();
    }

    sd_image_t* put() {
        if (image_.data != nullptr) {
            free(image_.data);
            image_.data = nullptr;
        }
        image_.width   = 0;
        image_.height  = 0;
        image_.channel = 0;
        return &image_;
    }

    sd_image_t& get() {
        return image_;
    }

    const sd_image_t& get() const {
        return image_;
    }

    sd_image_t release() {
        sd_image_t image = image_;
        image_           = {0, 0, 0, nullptr};
        return image;
    }

    void reset(sd_image_t image = {0, 0, 0, nullptr}) {
        if (image_.data != nullptr) {
            free(image_.data);
        }
        image_ = image;
    }
};

class SDAudioOwner {
private:
    uint32_t sample_rate_ = 0;
    uint32_t channels_    = 0;
    std::vector<float> samples_;

public:
    SDAudioOwner() = default;

    void reset(std::vector<float> samples = {}, uint32_t sample_rate = 0, uint32_t channels = 0) {
        samples_     = std::move(samples);
        sample_rate_ = sample_rate;
        channels_    = channels;
    }

    bool empty() const {
        return samples_.empty();
    }

    sd_audio_t get() {
        return {sample_rate_,
                channels_,
                channels_ == 0 ? 0 : static_cast<uint64_t>(samples_.size() / channels_),
                samples_.empty() ? nullptr : samples_.data()};
    }

    const std::vector<float>& samples() const {
        return samples_;
    }
};

class SDImageVec {
private:
    std::vector<sd_image_t> images_;
    std::vector<std::string> generation_infos_;

public:
    SDImageVec() = default;

    SDImageVec(const SDImageVec&)            = delete;
    SDImageVec& operator=(const SDImageVec&) = delete;

    SDImageVec(SDImageVec&& other) noexcept
        : images_(std::move(other.images_)), generation_infos_(std::move(other.generation_infos_)) {
    }

    SDImageVec& operator=(SDImageVec&& other) noexcept {
        if (this != &other) {
            clear();
            images_           = std::move(other.images_);
            generation_infos_ = std::move(other.generation_infos_);
        }
        return *this;
    }

    ~SDImageVec() {
        clear();
    }

    void push_back(sd_image_t image) {
        images_.push_back(image);
    }

    void push_back(SDImageOwner&& image) {
        images_.push_back(image.release());
    }

    void reserve(size_t count) {
        images_.reserve(count);
    }

    const std::string& generation_info(size_t index) const {
        static const std::string empty;
        return index < generation_infos_.size() ? generation_infos_[index] : empty;
    }

    void append_generation_info(size_t index, const std::string& info, const std::string& lora_model_dir = {}) {
        generation_infos_.resize(images_.size());
        generation_infos_.at(index) = merge_generation_info(generation_infos_.at(index), generation_info_with_relative_lora_paths(info, lora_model_dir));
    }

    void adopt(sd_image_generation_result_t* result, const std::string& lora_model_dir = {}) {
        clear();
        if (!result)
            return;
        for (int i = 0; i < result->count; ++i) {
            images_.push_back(result->items[i].image);
            result->items[i].image = {};
            generation_infos_.push_back(generation_info_with_relative_lora_paths(result->items[i].generation_info, lora_model_dir));
        }
        free_sd_image_generation_result(result);
    }

    void adopt_video(sd_video_generation_result_t* result, const std::string& lora_model_dir = {}) {
        clear();
        if (!result)
            return;
        const auto info = generation_info_with_relative_lora_paths(result->generation_info, lora_model_dir);
        for (int i = 0; i < result->frame_count; ++i) {
            images_.push_back(result->frames[i]);
            result->frames[i] = {};
            generation_infos_.push_back(info);
        }
        free_sd_video_generation_result(result);
    }

    void adopt(sd_image_t* images, int count) {
        clear();
        if (images == nullptr || count <= 0) {
            free(images);
            return;
        }

        images_.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            images_.push_back(images[i]);
        }
        free(images);
    }

    size_t size() const {
        return images_.size();
    }

    bool empty() const {
        return images_.empty();
    }

    int count() const {
        return static_cast<int>(images_.size());
    }

    explicit operator bool() const {
        return !images_.empty();
    }

    sd_image_t* data() {
        return images_.data();
    }

    const sd_image_t* data() const {
        return images_.data();
    }

    sd_image_t& operator[](size_t index) {
        return images_[index];
    }

    const sd_image_t& operator[](size_t index) const {
        return images_[index];
    }

    std::vector<sd_image_t>& raw() {
        return images_;
    }

    const std::vector<sd_image_t>& raw() const {
        return images_;
    }

    void clear() {
        for (sd_image_t& image : images_) {
            free(image.data);
            image.data = nullptr;
        }
        images_.clear();
        generation_infos_.clear();
    }
};

#endif  // __EXAMPLE_RESOURCE_OWNERS_H__

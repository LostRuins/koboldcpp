#ifndef __EXAMPLE_RESOURCE_OWNERS_H__
#define __EXAMPLE_RESOURCE_OWNERS_H__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

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

template <typename T>
using FreeUniquePtr = std::unique_ptr<T, FreeDeleter>;

using FilePtr        = std::unique_ptr<FILE, FileCloser>;
using SDCtxPtr       = std::unique_ptr<sd_ctx_t, SDCtxDeleter>;
using UpscalerCtxPtr = std::unique_ptr<upscaler_ctx_t, UpscalerCtxDeleter>;

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

class SDImageVec {
private:
    std::vector<sd_image_t> images_;

public:
    SDImageVec() = default;

    SDImageVec(const SDImageVec&)            = delete;
    SDImageVec& operator=(const SDImageVec&) = delete;

    SDImageVec(SDImageVec&& other) noexcept
        : images_(std::move(other.images_)) {
    }

    SDImageVec& operator=(SDImageVec&& other) noexcept {
        if (this != &other) {
            clear();
            images_ = std::move(other.images_);
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
    }
};

#endif  // __EXAMPLE_RESOURCE_OWNERS_H__

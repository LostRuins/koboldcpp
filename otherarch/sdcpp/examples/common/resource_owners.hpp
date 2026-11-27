#ifndef __EXAMPLE_RESOURCE_OWNERS_H__
#define __EXAMPLE_RESOURCE_OWNERS_H__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <optional>
#include "generation_metadata.h"
#include "log.h"
#include "stable-diffusion.h"

struct SDImage {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
    sd_tensor_t* storage = nullptr;
};

struct SDAudio {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    float* data;
    sd_tensor_t* storage = nullptr;
};

struct SDReferenceVideo {
    SDImage* frames;
    int frame_count;
    int fps;
    SDAudio audio;
};

struct SDTensorDeleter {
    void operator()(sd_tensor_t* tensor) const { sd_tensor_free(tensor); }
};
using SDTensorPtr = std::unique_ptr<sd_tensor_t, SDTensorDeleter>;

static inline SDImage tensor_image_view(const sd_tensor_t* tensor) {
    if (!tensor)
        return {};
    return {uint32_t(sd_tensor_dim(tensor, 1)), uint32_t(sd_tensor_dim(tensor, 2)), uint32_t(sd_tensor_dim(tensor, 0)),
            const_cast<uint8_t*>(static_cast<const uint8_t*>(sd_tensor_data(tensor)))};
}

static inline void sd_image_free(SDImage* image) {
    if (!image)
        return;
    if (image->storage)
        sd_tensor_free(image->storage);
    else
        free(image->data);
    *image = {};
}

static inline void free_sd_audio(SDAudio* audio) {
    if (!audio)
        return;
    if (audio->storage)
        sd_tensor_free(audio->storage);
    else
        free(audio->data);
    free(audio);
}

static inline SDTensorPtr image_tensor(const SDImage& image) {
    if (!image.data)
        return {};
    if (image.storage && sd_tensor_data(image.storage) == image.data &&
        sd_tensor_dim(image.storage, 0) == image.channel && sd_tensor_dim(image.storage, 1) == image.width && sd_tensor_dim(image.storage, 2) == image.height)
        return SDTensorPtr(sd_tensor_retain(image.storage));
    const int64_t shape[] = {image.channel, image.width, image.height};
    return SDTensorPtr(sd_tensor_create(SD_TENSOR_U8, 3, shape, image.data, size_t(image.width) * image.height * image.channel));
}

static inline SDTensorPtr audio_tensor(const SDAudio& audio) {
    if (!audio.data)
        return {};
    if (audio.storage)
        return SDTensorPtr(sd_tensor_retain(audio.storage));
    const int64_t shape[] = {audio.channels, static_cast<int64_t>(audio.sample_count)};
    return SDTensorPtr(sd_tensor_create(SD_TENSOR_F32, 2, shape, audio.data, size_t(audio.sample_count) * audio.channels * sizeof(float)));
}

static inline SDTensorPtr frames_tensor(const SDImage* frames, size_t count) {
    if (!frames || !count)
        return {};
    const auto& first        = frames[0];
    const size_t frame_bytes = size_t(first.width) * first.height * first.channel;
    if (!frame_bytes || count > SIZE_MAX / frame_bytes) {
        LOG_ERROR("invalid video frame storage size");
        return {};
    }
    std::vector<uint8_t> pixels(frame_bytes * count);
    for (size_t i = 0; i < count; ++i) {
        if (!frames[i].data || frames[i].width != first.width || frames[i].height != first.height || frames[i].channel != first.channel) {
            LOG_ERROR("video frames must have matching dimensions and channels");
            return {};
        }
        memcpy(pixels.data() + i * frame_bytes, frames[i].data, frame_bytes);
    }
    const int64_t shape[] = {first.channel, first.width, first.height, static_cast<int64_t>(count)};
    return SDTensorPtr(sd_tensor_create(SD_TENSOR_U8, 4, shape, pixels.data(), pixels.size()));
}

static inline SDAudio* retain_result_audio(const sd_result_t* result) {
    const auto* audio = sd_result_get_audio(result);
    if (!audio)
        return nullptr;
    auto* view = static_cast<SDAudio*>(malloc(sizeof(SDAudio)));
    if (!view)
        return nullptr;
    *view = {audio->sample_rate, uint32_t(sd_tensor_dim(audio->data, 0)), uint64_t(sd_tensor_dim(audio->data, 1)),
             const_cast<float*>(static_cast<const float*>(sd_tensor_data(audio->data))), sd_tensor_retain(audio->data)};
    return view;
}

static inline bool preprocess_canny(SDImage& image, float high, float low, float weak, float strong, bool inverse) {
    auto input          = image_tensor(image);
    sd_tensor_t* output = nullptr;
    if (!input || sd_preprocess_canny(input.get(), high, low, weak, strong, inverse, &output) != SD_STATUS_OK)
        return false;
    sd_image_free(&image);
    image         = tensor_image_view(output);
    image.storage = output;
    return true;
}

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
            sd_ctx_free(ctx);
        }
    }
};

struct UpscalerCtxDeleter {
    void operator()(sd_upscaler_t* ctx) const {
        if (ctx != nullptr) {
            sd_upscaler_free(ctx);
        }
    }
};

struct ADetailerCtxDeleter {
    void operator()(sd_adetailer_t* ctx) const {
        if (ctx != nullptr) {
            sd_adetailer_free(ctx);
        }
    }
};

template <typename T>
using FreeUniquePtr = std::unique_ptr<T, FreeDeleter>;

using FilePtr         = std::unique_ptr<FILE, FileCloser>;
using SDCtxPtr        = std::unique_ptr<sd_ctx_t, SDCtxDeleter>;
using UpscalerCtxPtr  = std::unique_ptr<sd_upscaler_t, UpscalerCtxDeleter>;
using ADetailerCtxPtr = std::unique_ptr<sd_adetailer_t, ADetailerCtxDeleter>;

struct SDOptDeleter {
    void operator()(sd_opt_t* opt) const { sd_opt_free(opt); }
};
struct SDResultDeleter {
    void operator()(sd_result_t* result) const { sd_result_free(result); }
};
using SDOptPtr    = std::unique_ptr<sd_opt_t, SDOptDeleter>;
using SDResultPtr = std::unique_ptr<sd_result_t, SDResultDeleter>;

class SDOptBuilder {
    SDOptPtr opt;
    sd_status_t status;

public:
    explicit SDOptBuilder(sd_opt_t* value)
        : opt(value), status(value ? SD_STATUS_OK : SD_STATUS_OUT_OF_MEMORY) {}
    SDOptPtr finish() { return status == SD_STATUS_OK ? std::move(opt) : SDOptPtr{}; }
    void unset(const char* key) {
        if (status == SD_STATUS_OK)
            status = sd_opt_unset(opt.get(), key);
    }
    void set_floats(const char* key, const std::optional<std::vector<float>>& value) {
        if (value)
            set_floats(key, value->data(), value->size());
    }
    void set_bool(const char* key, const std::optional<bool>& value) {
        if (value)
            set_bool(key, *value);
    }
    template <typename T>
    void set_int(const char* key, const std::optional<T>& value) {
        if (value)
            set_int(key, *value);
    }
    template <typename T>
    void set_float(const char* key, const std::optional<T>& value) {
        if (value)
            set_float(key, *value);
    }
    void set_string(const char* key, const std::optional<std::string>& value) {
        if (value)
            set_string(key, *value);
    }
    void set_bool(const char* key, bool value) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_bool(opt.get(), key, value);
    }
    void set_int(const char* key, int64_t value) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_int(opt.get(), key, value);
    }
    void set_float(const char* key, double value) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_float(opt.get(), key, value);
    }
    void set_string(const char* key, const std::string& value) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_string(opt.get(), key, value.c_str());
    }
    void set_string(const char* key, const char* value) {
        if (value && status == SD_STATUS_OK)
            status = sd_opt_set_string(opt.get(), key, value);
    }
    void set_image(const char* key, const SDImage& value) {
        if (!value.data || status != SD_STATUS_OK)
            return;
        auto tensor = image_tensor(value);
        if (!tensor) {
            status = SD_STATUS_INVALID_ARGUMENT;
            return;
        }
        status = sd_opt_set_tensor(opt.get(), key, tensor.get());
    }
    void set_images(const char* key, const SDImage* value, size_t count) {
        if (status != SD_STATUS_OK)
            return;
        std::vector<SDTensorPtr> storage;
        std::vector<const sd_tensor_t*> tensors;
        storage.reserve(count);
        tensors.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            storage.push_back(image_tensor(value[i]));
            if (!storage.back()) {
                status = SD_STATUS_INVALID_ARGUMENT;
                return;
            }
            tensors.push_back(storage.back().get());
        }
        status = sd_opt_set_tensors(opt.get(), key, tensors.data(), count);
    }
    void set_frames(const char* key, const SDImage* value, size_t count) {
        if (!count || status != SD_STATUS_OK)
            return;
        auto tensor = frames_tensor(value, count);
        if (!tensor) {
            status = SD_STATUS_INVALID_ARGUMENT;
            return;
        }
        status = sd_opt_set_tensor(opt.get(), key, tensor.get());
    }
    void set_floats(const char* key, const float* value, size_t count) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_floats(opt.get(), key, value, count);
    }
    void set_ints(const char* key, const int32_t* value, size_t count) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_ints(opt.get(), key, value, count);
    }
    void set_loras(const char* key, const sd_lora_t* value, size_t count) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_loras(opt.get(), key, value, count);
    }
    void set_embeddings(const char* key, const sd_embedding_t* value, size_t count) {
        if (status == SD_STATUS_OK)
            status = sd_opt_set_embeddings(opt.get(), key, value, count);
    }
    void set_audios(const char* key, const SDAudio* value, size_t count) {
        if (status != SD_STATUS_OK)
            return;
        std::vector<SDTensorPtr> storage;
        std::vector<sd_audio_t> audios;
        storage.reserve(count);
        audios.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            storage.push_back(audio_tensor(value[i]));
            audios.push_back({value[i].sample_rate, storage.back().get()});
        }
        status = sd_opt_set_audios(opt.get(), key, audios.data(), count);
    }
    void set_videos(const char* key, const SDReferenceVideo* value, size_t count) {
        if (status != SD_STATUS_OK)
            return;
        std::vector<SDTensorPtr> storage;
        std::vector<sd_video_t> videos;
        storage.reserve(count * 2);
        videos.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            storage.push_back(frames_tensor(value[i].frames, value[i].frame_count));
            const auto* frames = storage.back().get();
            storage.push_back(audio_tensor(value[i].audio));
            videos.push_back({frames, value[i].fps, {value[i].audio.sample_rate, storage.back().get()}});
        }
        status = sd_opt_set_videos(opt.get(), key, videos.data(), count);
    }
};

static inline sd_ctx_t* create_sd_context(const SDOptPtr& opt) {
    sd_ctx_t* ctx = nullptr;
    if (opt)
        sd_ctx_create(opt.get(), &ctx);
    return ctx;
}

static inline bool generate_sd_image(sd_ctx_t* ctx, const SDOptPtr& opt, sd_result_t** result) {
    if (result)
        *result = nullptr;
    return opt && sd_generate_image(ctx, opt.get(), result) == SD_STATUS_OK;
}

static inline bool generate_sd_video(sd_ctx_t* ctx, const SDOptPtr& opt, sd_result_t** result) {
    if (result)
        *result = nullptr;
    return opt && sd_generate_video(ctx, opt.get(), result) == SD_STATUS_OK;
}

static inline sd_upscaler_t* create_sd_upscaler(const char* path, std::optional<bool> direct, std::optional<int> threads, std::optional<int> tile, const char* backend, const char* params_backend) {
    SDOptPtr opt(sd_upscaler_opt_create());
    if (!opt || sd_opt_set_string(opt.get(), "model_path", path) != SD_STATUS_OK ||
        (direct && sd_opt_set_bool(opt.get(), "conv_direct", *direct) != SD_STATUS_OK) ||
        (tile && sd_opt_set_int(opt.get(), "tile_size", *tile) != SD_STATUS_OK) ||
        (threads && sd_opt_set_int(opt.get(), "n_threads", *threads) != SD_STATUS_OK) ||
        (backend && sd_opt_set_string(opt.get(), "backend", backend) != SD_STATUS_OK) ||
        (params_backend && sd_opt_set_string(opt.get(), "params_backend", params_backend) != SD_STATUS_OK))
        return nullptr;
    sd_upscaler_t* result = nullptr;
    sd_upscaler_create(opt.get(), &result);
    return result;
}

static inline bool upscale_sd_image(sd_upscaler_t* ctx, SDImage image, sd_result_t** result) {
    if (result)
        *result = nullptr;
    SDOptPtr opt(sd_upscale_opt_create());
    auto tensor = image_tensor(image);
    return opt && tensor && sd_opt_set_tensor(opt.get(), "image", tensor.get()) == SD_STATUS_OK &&
           sd_upscale(ctx, opt.get(), result) == SD_STATUS_OK;
}

static inline sd_adetailer_t* create_sd_adetailer(const char* path, std::optional<int> threads, const char* backend, const char* params_backend) {
    SDOptPtr opt(sd_adetailer_opt_create());
    if (!opt || sd_opt_set_string(opt.get(), "model_path", path) != SD_STATUS_OK ||
        (threads && sd_opt_set_int(opt.get(), "n_threads", *threads) != SD_STATUS_OK) ||
        (backend && sd_opt_set_string(opt.get(), "backend", backend) != SD_STATUS_OK) ||
        (params_backend && sd_opt_set_string(opt.get(), "params_backend", params_backend) != SD_STATUS_OK))
        return nullptr;
    sd_adetailer_t* result = nullptr;
    sd_adetailer_create(opt.get(), &result);
    return result;
}

static inline bool adetail_sd_image(sd_adetailer_t* detector, sd_ctx_t* ctx, SDImage image, SDOptPtr opt, sd_result_t** result) {
    if (result)
        *result = nullptr;
    auto tensor = image_tensor(image);
    return opt && tensor && sd_opt_set_tensor(opt.get(), "image", tensor.get()) == SD_STATUS_OK &&
           sd_adetail(detector, ctx, opt.get(), result) == SD_STATUS_OK;
}

struct SDVideoUpscalerParams {
    const char* diffusion_model_path{};
    const char* vae_path{};
    // Explicit output scale; RelayVSR requires 2 or 4 matching both checkpoints.
    int upscale_factor{};
    std::optional<int> n_threads;
    const char* backend{};
    const char* params_backend{};
    // auto/on/off or diffusion/vae assignments; null selects auto.
    const char* flash_attn_mode{};
    const char* conv_direct_mode{};
};

struct SDVideoUpscaleParams {
    const SDImage* frames{};
    int frame_count{};
    std::optional<int> fps;
    std::optional<int64_t> seed;
    // RelayVSR reference window, including both endpoints; defaults to 16.
    std::optional<int> window_size;
    // SeedVR2 independent temporal chunks in input frames (1 or 4n+1); defaults to 5.
    std::optional<int> temporal_chunk_size;
};

static inline sd_vid_upscaler_t* create_sd_vid_upscaler(const SDVideoUpscalerParams& config) {
    SDOptPtr opt(sd_vid_upscaler_opt_create());
    if (!opt || sd_opt_set_string(opt.get(), "diffusion_model_path", config.diffusion_model_path) != SD_STATUS_OK ||
        sd_opt_set_string(opt.get(), "vae_path", config.vae_path) != SD_STATUS_OK ||
        sd_opt_set_int(opt.get(), "upscale_factor", config.upscale_factor) != SD_STATUS_OK ||
        (config.n_threads && sd_opt_set_int(opt.get(), "n_threads", *config.n_threads) != SD_STATUS_OK) ||
        (config.backend && sd_opt_set_string(opt.get(), "backend", config.backend) != SD_STATUS_OK) ||
        (config.params_backend && sd_opt_set_string(opt.get(), "params_backend", config.params_backend) != SD_STATUS_OK) ||
        (config.flash_attn_mode && sd_opt_set_string(opt.get(), "flash_attn_mode", config.flash_attn_mode) != SD_STATUS_OK) ||
        (config.conv_direct_mode && sd_opt_set_string(opt.get(), "conv_direct_mode", config.conv_direct_mode) != SD_STATUS_OK))
        return nullptr;
    sd_vid_upscaler_t* result = nullptr;
    sd_vid_upscaler_create(opt.get(), &result);
    return result;
}

static inline bool upscale_sd_video(sd_vid_upscaler_t* ctx, const SDVideoUpscaleParams& config, sd_result_t** result) {
    if (result)
        *result = nullptr;
    SDOptBuilder opt(sd_vid_upscale_opt_create());
    opt.set_frames("frames", config.frames, config.frame_count);
    opt.set_int("fps", config.fps);
    opt.set_int("seed", config.seed);
    opt.set_int("window_size", config.window_size);
    opt.set_int("temporal_chunk_size", config.temporal_chunk_size);
    auto params = opt.finish();
    return params && sd_upscale_video(ctx, params.get(), result) == SD_STATUS_OK;
}

static inline SDImage take_result_image(sd_result_t* result, size_t index) {
    const auto* tensor = sd_result_get_image(result, index);
    auto image         = tensor_image_view(tensor);
    image.storage      = sd_tensor_retain(tensor);
    return image;
}

class SDImageOwner {
private:
    static SDImage copy_image(const SDImage& image) {
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

    SDImage image_ = {0, 0, 0, nullptr};
    SDTensorPtr public_tensor_;

public:
    SDImageOwner() = default;
    explicit SDImageOwner(SDImage image)
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

    const sd_tensor_t* tensor() {
        public_tensor_ = image_tensor(image_);
        return public_tensor_.get();
    }

    SDImage* put() {
        if (image_.data != nullptr) {
            sd_image_free(&image_);
            image_.data = nullptr;
        }
        image_.width   = 0;
        image_.height  = 0;
        image_.channel = 0;
        return &image_;
    }

    SDImage& get() {
        return image_;
    }

    const SDImage& get() const {
        return image_;
    }

    SDImage release() {
        SDImage image = image_;
        image_        = {0, 0, 0, nullptr};
        return image;
    }

    void reset(SDImage image = {0, 0, 0, nullptr}) {
        if (image_.data != nullptr) {
            sd_image_free(&image_);
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

    SDAudio get() {
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
    std::vector<SDImage> images_;
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

    void push_back(SDImage image) {
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

    void adopt(sd_result_t* result, const std::string& lora_model_dir = {}) {
        clear();
        SDResultPtr owner(result);
        if (!result)
            return;
        const size_t count = sd_result_get_image_count(result);
        images_.reserve(count);
        generation_infos_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            generation_infos_.push_back(generation_info_with_relative_lora_paths(sd_result_get_generation_info(result, i), lora_model_dir));
            images_.push_back(take_result_image(result, i));
        }
    }

    void adopt_video(sd_result_t* result, const std::string& lora_model_dir = {}) {
        adopt(result, lora_model_dir);
    }

    void adopt(SDImage* images, int count) {
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

    SDImage* data() {
        return images_.data();
    }

    const SDImage* data() const {
        return images_.data();
    }

    SDImage& operator[](size_t index) {
        return images_[index];
    }

    const SDImage& operator[](size_t index) const {
        return images_[index];
    }

    std::vector<SDImage>& raw() {
        return images_;
    }

    const std::vector<SDImage>& raw() const {
        return images_;
    }

    void clear() {
        for (SDImage& image : images_) {
            sd_image_free(&image);
            image.data = nullptr;
        }
        images_.clear();
        generation_infos_.clear();
    }
};

#endif  // __EXAMPLE_RESOURCE_OWNERS_H__

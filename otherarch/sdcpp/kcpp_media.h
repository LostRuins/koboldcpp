// kcpp_media.h — read-only media views over shared storage (pixels, audio
// samples) for the sdcpp boundary. Owned by Koboldcpp.
//
// A KcppImage owns its pixels (HWC, interleaved uint8) and a KcppAudio owns
// its samples (interleaved float) through a shared_ptr, and both hand out
// plain sd_image_t / sd_audio_t views for the sdcpp boundary. All operations
// are read-only: derived media (resize, flipped_mask) are new objects and
// never mutate their source. Views borrow the storage; they are valid while
// the owning media object is alive.
//
// An image is either null (w/h/c == 0, px == nullptr — the default state and
// the result of a failed decode) or valid (w,h,c > 0, px non-null with
// px->size() == w*h*c). An audio is null the same way (sample_rate/channels
// == 0, samples == nullptr) or valid (both > 0, samples non-empty with
// samples->size() a multiple of channels).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "stable-diffusion.h"
#include "otherarch/utils.h"      // kcpp_base64_decode
#include "otherarch/webp_image.h" // kcpp_decode_webp_from_memory
#include "stb_image.h"
#include "stb_image_resize.h"

struct KcppImage {
    uint32_t w = 0, h = 0, c = 0;
    std::shared_ptr<const std::vector<uint8_t>> px;   // HWC, interleaved

    bool valid() const { return w > 0 && h > 0 && c > 0 && px != nullptr; }

    // decode only (always 3 channels, any source channel count), no fit;
    // null image on failure
    static KcppImage from_b64(const std::string& b64);

    // old API bridge: copy of a vendored sd_image_t (replaced by from_tensor
    // once the sdcpp results move to tensors); null image on an invalid source
    static KcppImage from_sd_image(const sd_image_t&);

    // center-crop to the target aspect, resize to exact w/h, convert to c
    // channels; short-circuits when the target already matches; null image
    // on failure
    KcppImage resize(uint32_t w, uint32_t h, uint32_t c) const;

    // per-pixel inversion (255 - v); 1-channel images only
    KcppImage flipped_mask() const;

    // fill all channels with fill
    static KcppImage solid(uint32_t w, uint32_t h, uint32_t c, uint8_t fill);

    // true when dimensions and pixel content match (two null images match)
    bool same_as(const KcppImage& other) const;

    // old API boundary: borrows the pixels; a null image views as {0,0,0,nullptr}
    sd_image_t view() const;

    // copy count pixels, converting between channel counts (the vendored
    // stb_image_resize only handles a single channel count); false when the
    // conversion is unsupported
    static bool convert_pixels(const uint8_t* src, uint32_t src_c,
                               uint8_t* dst, uint32_t dst_c, size_t count);

    /* future api
    std::shared_ptr<const std::vector<uint8_t>> to_planar() const;  // sd_tensor_create
    static KcppImage from_tensor(const sd_tensor_t*); // results/previews (copies)
    */
};

inline KcppImage KcppImage::from_b64(const std::string& b64)
{
    const std::vector<uint8_t> decoded = kcpp_base64_decode(b64);
    int w = 0, h = 0, c = 0;
    uint8_t* raw = stbi_load_from_memory(decoded.data(), decoded.size(), &w, &h, &c, 3);
    if (raw == nullptr) {
        // stbi has no webp support: fall back to the kcpp webp decoder
        raw = kcpp_decode_webp_from_memory(decoded.data(), decoded.size(), &w, &h, &c, 3);
    }
    if (raw == nullptr || w <= 0 || h <= 0) {
        stbi_image_free(raw);
        fprintf(stderr, "KcppImage::from_b64: failed to decode image\n");
        return KcppImage{};
    }
    // both decoders are asked for 3 channels (stbi also reports the channel
    // count of the file in c, which is not the count of the returned pixels)
    const uint32_t channels = 3;
    auto v = std::make_shared<std::vector<uint8_t>>(raw, raw + (size_t)w * h * channels);
    stbi_image_free(raw);
    return KcppImage{(uint32_t)w, (uint32_t)h, channels, std::move(v)};
}

inline KcppImage KcppImage::from_sd_image(const sd_image_t& image)
{
    if (image.width == 0 || image.height == 0 || image.channel == 0 || image.data == nullptr) {
        return KcppImage{};
    }
    auto v = std::make_shared<std::vector<uint8_t>>(
        image.data, image.data + (size_t)image.width * image.height * image.channel);
    return KcppImage{image.width, image.height, image.channel, std::move(v)};
}

inline KcppImage KcppImage::resize(uint32_t w, uint32_t h, uint32_t c) const
{
    if (!valid()) {
        return KcppImage{};
    }
    if (w == this->w && h == this->h && c == this->c) {
        return *this;
    }

    // center-crop to the target aspect
    uint32_t sw = this->w, sh = this->h;
    uint32_t sx = 0, sy = 0;
    const float dst_aspect = (float)w / (float)h;
    const float src_aspect = (float)sw / (float)sh;
    if (src_aspect > dst_aspect) {
        sw = (uint32_t)((float)sh * dst_aspect);
        sx = (this->w - sw) / 2;
    } else if (src_aspect < dst_aspect) {
        sh = (uint32_t)((float)sw / dst_aspect);
        sy = (this->h - sh) / 2;
    }

    // copy the cropped region, converting to the target channel count on the
    // way (so the conversion happens at the source resolution, as it did when
    // stb_image decoded directly at c channels)
    std::vector<uint8_t> source((size_t)sw * sh * c);
    for (uint32_t y = 0; y < sh; ++y) {
        if (!convert_pixels(px->data() + ((size_t)(sy + y) * this->w + sx) * this->c,
                            this->c, source.data() + (size_t)y * sw * c, c, sw)) {
            fprintf(stderr, "KcppImage::resize: unsupported %u -> %u channel conversion\n",
                    this->c, c);
            return KcppImage{};
        }
    }
    if (sw == w && sh == h) {
        auto v = std::make_shared<std::vector<uint8_t>>(std::move(source));
        return KcppImage{w, h, c, std::move(v)};
    }

    auto out = std::make_shared<std::vector<uint8_t>>((size_t)w * h * c);
    if (!stbir_resize(source.data(), sw, sh, (int)(sw * c),
                      out->data(), w, h, (int)(w * c),
                      STBIR_TYPE_UINT8, (int)c, STBIR_ALPHA_CHANNEL_NONE, 0,
                      STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                      STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                      STBIR_COLORSPACE_SRGB, nullptr)) {
        fprintf(stderr, "KcppImage::resize: stbir_resize failed\n");
        return KcppImage{};
    }
    return KcppImage{w, h, c, std::move(out)};
}

inline bool KcppImage::convert_pixels(const uint8_t* src, uint32_t src_c,
                                       uint8_t* dst, uint32_t dst_c, size_t count)
{
    if (src_c == dst_c) {
        std::memcpy(dst, src, count * dst_c);
        return true;
    }
    if (src_c < 1 || src_c > 4 || dst_c < 1 || dst_c > 4) {
        return false;
    }
    // 2-channel input (gray + alpha) has no meaningful conversion here
    if (src_c == 2) {
        return false;
    }
    for (size_t i = 0; i < count; ++i, src += src_c, dst += dst_c) {
        switch (dst_c) {
        case 1:
            // same luma as stb_image's own N -> 1 conversion
            dst[0] = src_c == 1 ? src[0]
                                : (uint8_t)(((src[0] * 77) + (src[1] * 150) + (src[2] * 29)) >> 8);
            break;
        case 3:
            dst[0] = src[0];
            dst[1] = src_c == 1 ? src[0] : src[1];
            dst[2] = src_c == 1 ? src[0] : src[2];
            break;
        case 4:
            dst[0] = src[0];
            dst[1] = src_c == 1 ? src[0] : src[1];
            dst[2] = src_c == 1 ? src[0] : src[2];
            dst[3] = src_c == 4 ? src[3] : 255;
            break;
        default:
            return false;
        }
    }
    return true;
}

inline KcppImage KcppImage::flipped_mask() const
{
    if (!valid() || c != 1) {
        return KcppImage{};
    }
    auto v = std::make_shared<std::vector<uint8_t>>(*px);
    for (uint8_t& b : *v) {
        b = 255 - b;
    }
    return KcppImage{w, h, c, std::move(v)};
}

inline KcppImage KcppImage::solid(uint32_t w, uint32_t h, uint32_t c, uint8_t fill)
{
    auto v = std::make_shared<std::vector<uint8_t>>((size_t)w * h * c, fill);
    return KcppImage{w, h, c, std::move(v)};
}

inline bool KcppImage::same_as(const KcppImage& other) const
{
    if (!valid() || !other.valid()) {
        return !valid() && !other.valid();
    }
    return w == other.w && h == other.h && c == other.c && *px == *other.px;
}

inline sd_image_t KcppImage::view() const
{
    if (!valid()) {
        return sd_image_t{0, 0, 0, nullptr};
    }
    return sd_image_t{w, h, c, const_cast<uint8_t*>(px->data())};
}

// a list of KcppImages (e.g. reference images)
struct KcppImages {
    KcppImages() = default;
    // copied views hold pointers into the source's pixel storage: rebuild
    KcppImages(const KcppImages& other) : items_(other.items_), views_stale_(true) {}
    KcppImages& operator=(const KcppImages& other) {
        items_ = other.items_;
        views_stale_ = true;
        return *this;
    }

    bool empty() const { return items_.empty(); }
    size_t size() const { return items_.size(); }
    const KcppImage& at(size_t i) const { return items_[i]; }
    const KcppImage& operator[](size_t i) const { return items_[i]; }
    void push_back(const KcppImage& img) { items_.push_back(img); views_stale_ = true; }
    void clear() { items_.clear(); views_stale_ = true; }

    // old API boundary: views owned by *this, invalidated by push_back/clear
    // (views are only needed while the sdcpp call is in flight)
    sd_image_t* views() const {
        if (views_stale_) {
            views_.resize(items_.size());
            for (size_t i = 0; i < items_.size(); ++i) {
                views_[i] = items_[i].view();
            }
            views_stale_ = false;
        }
        return views_.data();
    }

private:
    std::vector<KcppImage> items_;
    mutable std::vector<sd_image_t> views_;
    mutable bool views_stale_ = true;
};

// an audio track: interleaved float PCM plus sample-rate/channel metadata
struct KcppAudio {
    uint32_t sample_rate = 0, channels = 0;
    std::shared_ptr<const std::vector<float>> samples;   // interleaved

    bool valid() const {
        return sample_rate > 0 && channels > 0 && samples != nullptr && !samples->empty();
    }

    // base64 audio file (any format the kcpp decoder reads) at its natural
    // rate/channels, no resampling; null audio on failure
    static KcppAudio from_b64(const std::string& b64audio);

    // old API bridge: copy of a vendored sd_audio_t (interleaved; replaced by
    // from_audio once the sdcpp results move to tensors); null audio on an
    // invalid source
    static KcppAudio from_sd_audio(const sd_audio_t&);

    // old API boundary: borrows the samples; a null audio views as {0,0,0,nullptr}
    sd_audio_t view() const;

    /* future api
    std::shared_ptr<const std::vector<float>> to_planar() const;  // sd_tensor_create
    static KcppAudio from_tensor(const sd_audio_t*);              // results (copies)
    */
};

inline KcppAudio KcppAudio::from_b64(const std::string& b64audio)
{
    if (b64audio.empty()) {
        return KcppAudio{};
    }
    const std::vector<uint8_t> decoded = kcpp_base64_decode(b64audio);
    std::vector<float> pcm;
    int sample_rate = 0;
    int channels = 0;
    if (!kcpp_decode_audio_file_from_buf(decoded.data(), decoded.size(),
                                         sample_rate, channels, pcm) ||
        sample_rate <= 0 || channels <= 0 || pcm.empty()) {
        fprintf(stderr, "KcppAudio::from_b64: failed to decode audio\n");
        return KcppAudio{};
    }
    // a partial trailing frame carries no usable sample: drop it, so the view's
    // sample_count * channels covers the whole buffer
    if (pcm.size() % (size_t)channels != 0) {
        pcm.resize(pcm.size() - pcm.size() % (size_t)channels);
    }
    auto v = std::make_shared<std::vector<float>>(std::move(pcm));
    return KcppAudio{(uint32_t)sample_rate, (uint32_t)channels, std::move(v)};
}

inline KcppAudio KcppAudio::from_sd_audio(const sd_audio_t& audio)
{
    if (audio.sample_rate == 0 || audio.channels == 0 || audio.sample_count == 0 || audio.data == nullptr) {
        return KcppAudio{};
    }
    const size_t count = (size_t)audio.sample_count * audio.channels;
    auto v = std::make_shared<std::vector<float>>(audio.data, audio.data + count);
    return KcppAudio{audio.sample_rate, audio.channels, std::move(v)};
}

inline sd_audio_t KcppAudio::view() const
{
    if (!valid()) {
        return sd_audio_t{0, 0, 0, nullptr};
    }
    return sd_audio_t{sample_rate, channels,
                      (uint64_t)(samples->size() / channels),
                      const_cast<float*>(samples->data())};
}

// a list of KcppAudios (e.g. reference audios)
struct KcppAudios {
    KcppAudios() = default;
    // copied views hold pointers into the source's sample storage: rebuild
    KcppAudios(const KcppAudios& other) : items_(other.items_), views_stale_(true) {}
    KcppAudios& operator=(const KcppAudios& other) {
        items_ = other.items_;
        views_stale_ = true;
        return *this;
    }

    bool empty() const { return items_.empty(); }
    size_t size() const { return items_.size(); }
    const KcppAudio& at(size_t i) const { return items_[i]; }
    const KcppAudio& operator[](size_t i) const { return items_[i]; }
    void push_back(const KcppAudio& audio) { items_.push_back(audio); views_stale_ = true; }
    void clear() { items_.clear(); views_stale_ = true; }

    // old API boundary: views owned by *this, invalidated by push_back/clear
    // (views are only needed while the sdcpp call is in flight)
    sd_audio_t* views() const {
        if (views_stale_) {
            views_.resize(items_.size());
            for (size_t i = 0; i < items_.size(); ++i) {
                views_[i] = items_[i].view();
            }
            views_stale_ = false;
        }
        return views_.data();
    }

private:
    std::vector<KcppAudio> items_;
    mutable std::vector<sd_audio_t> views_;
    mutable bool views_stale_ = true;
};

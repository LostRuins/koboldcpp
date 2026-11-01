// Project-owned bridge between stb_image callers and the vendored WebP decoder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "webp/decode.h"

static inline bool kcpp_is_webp_image(const uint8_t * data, size_t size) {
    return data != nullptr && size >= 12 &&
           data[0] == 'R' && data[1] == 'I' &&
           data[2] == 'F' && data[3] == 'F' &&
           data[8] == 'W' && data[9] == 'E' &&
           data[10] == 'B' && data[11] == 'P';
}

// Decodes a static WebP into malloc-owned interleaved pixels. The interface and
// ownership intentionally match the subset of stbi_load_from_memory used by
// KoboldCpp, so callers can continue to release the result with
// stbi_image_free(). Animated WebP is deliberately left to animation-aware
// callers.
static inline uint8_t * kcpp_decode_webp_from_memory(
        const uint8_t * data, size_t size,
        int * width, int * height, int * channels_in_file,
        int desired_channels) {
    if (!kcpp_is_webp_image(data, size)) {
        return nullptr;
    }

    WebPBitstreamFeatures features = {};
    if (WebPGetFeatures(data, size, &features) != VP8_STATUS_OK ||
        features.has_animation || features.width <= 0 || features.height <= 0) {
        return nullptr;
    }

    const int source_channels = features.has_alpha ? 4 : 3;
    const int output_channels = desired_channels == 0 ? source_channels : desired_channels;
    if (output_channels != 1 && output_channels != 3 && output_channels != 4) {
        return nullptr;
    }

    const size_t image_width = static_cast<size_t>(features.width);
    const size_t image_height = static_cast<size_t>(features.height);
    if (image_width > std::numeric_limits<size_t>::max() / image_height) {
        return nullptr;
    }
    const size_t pixel_count = image_width * image_height;
    if (pixel_count > std::numeric_limits<size_t>::max() /
                          static_cast<size_t>(output_channels)) {
        return nullptr;
    }

    uint8_t * output = static_cast<uint8_t *>(
        std::malloc(pixel_count * static_cast<size_t>(output_channels)));
    if (output == nullptr) {
        return nullptr;
    }

    bool decoded = false;
    if (output_channels == 1) {
        if (pixel_count <= std::numeric_limits<size_t>::max() / 3) {
            const size_t rgb_size = pixel_count * 3;
            uint8_t * rgb = static_cast<uint8_t *>(std::malloc(rgb_size));
            if (rgb != nullptr &&
                WebPDecodeRGBInto(data, size, rgb, rgb_size, features.width * 3) != nullptr) {
                for (size_t i = 0; i < pixel_count; ++i) {
                    output[i] = static_cast<uint8_t>(
                        (77 * rgb[3 * i] + 150 * rgb[3 * i + 1] +
                         29 * rgb[3 * i + 2] + 128) >> 8);
                }
                decoded = true;
            }
            std::free(rgb);
        }
    } else if (output_channels == 3) {
        decoded = WebPDecodeRGBInto(
            data, size, output, pixel_count * 3, features.width * 3) != nullptr;
    } else {
        decoded = WebPDecodeRGBAInto(
            data, size, output, pixel_count * 4, features.width * 4) != nullptr;
    }

    if (!decoded) {
        std::free(output);
        return nullptr;
    }

    if (width != nullptr) {
        *width = features.width;
    }
    if (height != nullptr) {
        *height = features.height;
    }
    if (channels_in_file != nullptr) {
        *channels_in_file = source_channels;
    }
    return output;
}

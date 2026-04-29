#include "media_io.h"
#include "log.h"
#include "resource_owners.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#ifdef SD_USE_WEBP
#include "webp/decode.h"
#include "webp/encode.h"
#include "webp/mux.h"
#endif

#ifdef SD_USE_WEBM
#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvwriter.h"
#endif

namespace fs = std::filesystem;

#ifdef SD_USE_WEBP
struct WebPFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            WebPFree(ptr);
        }
    }
};

struct WebPMuxDeleter {
    void operator()(WebPMux* mux) const {
        if (mux != nullptr) {
            WebPMuxDelete(mux);
        }
    }
};

struct WebPAnimEncoderDeleter {
    void operator()(WebPAnimEncoder* enc) const {
        if (enc != nullptr) {
            WebPAnimEncoderDelete(enc);
        }
    }
};

struct WebPDataGuard {
    WebPDataGuard() {
        WebPDataInit(&data);
    }

    ~WebPDataGuard() {
        WebPDataClear(&data);
    }

    WebPData data;
};

struct WebPPictureGuard {
    WebPPictureGuard()
        : initialized(WebPPictureInit(&picture) != 0) {
    }

    ~WebPPictureGuard() {
        if (initialized) {
            WebPPictureFree(&picture);
        }
    }

    WebPPicture picture;
    bool initialized;
};

using WebPBufferPtr      = std::unique_ptr<uint8_t, WebPFreeDeleter>;
using WebPMuxPtr         = std::unique_ptr<WebPMux, WebPMuxDeleter>;
using WebPAnimEncoderPtr = std::unique_ptr<WebPAnimEncoder, WebPAnimEncoderDeleter>;
#endif

#ifdef SD_USE_WEBM
class MemoryMkvWriter : public mkvmuxer::IMkvWriter {
public:
    mkvmuxer::int32 Write(const void* buf, mkvmuxer::uint32 len) override {
        if (buf == nullptr && len > 0) {
            return -1;
        }
        const size_t end_pos = position_ + static_cast<size_t>(len);
        if (end_pos > data_.size()) {
            data_.resize(end_pos);
        }
        if (len > 0) {
            memcpy(data_.data() + position_, buf, len);
        }
        position_ = end_pos;
        return 0;
    }

    mkvmuxer::int64 Position() const override {
        return static_cast<mkvmuxer::int64>(position_);
    }

    mkvmuxer::int32 Position(mkvmuxer::int64 position) override {
        if (position < 0) {
            return -1;
        }
        const size_t target = static_cast<size_t>(position);
        if (target > data_.size()) {
            data_.resize(target);
        }
        position_ = target;
        return 0;
    }

    bool Seekable() const override {
        return true;
    }

    void ElementStartNotify(mkvmuxer::uint64, mkvmuxer::int64) override {
    }

    const std::vector<uint8_t>& data() const {
        return data_;
    }

private:
    std::vector<uint8_t> data_;
    size_t position_ = 0;
};
#endif

bool read_binary_file_bytes(const char* path, std::vector<uint8_t>& data) {
    std::ifstream fin(fs::path(path), std::ios::binary);
    if (!fin) {
        return false;
    }

    fin.seekg(0, std::ios::end);
    std::streampos size = fin.tellg();
    if (size < 0) {
        return false;
    }
    fin.seekg(0, std::ios::beg);

    data.resize(static_cast<size_t>(size));
    if (!data.empty()) {
        fin.read(reinterpret_cast<char*>(data.data()), size);
        if (!fin) {
            return false;
        }
    }
    return true;
}

bool write_binary_file_bytes(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream fout(fs::path(path), std::ios::binary);
    if (!fout) {
        return false;
    }

    if (!data.empty()) {
        fout.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!fout) {
            return false;
        }
    }
    return true;
}

uint32_t read_u32_le_bytes(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

int stbi_ext_write_png_to_func(stbi_write_func* func,
                               void* context,
                               int x,
                               int y,
                               int comp,
                               const void* data,
                               int stride_bytes,
                               const char* parameters) {
    int len            = 0;
    unsigned char* png = stbi_write_png_to_mem((const unsigned char*)data, stride_bytes, x, y, comp, &len, parameters);
    if (png == nullptr) {
        return 0;
    }
    func(context, png, len);
    STBIW_FREE(png);
    return 1;
}

bool is_webp_signature(const uint8_t* data, size_t size) {
    return size >= 12 &&
           memcmp(data, "RIFF", 4) == 0 &&
           memcmp(data + 8, "WEBP", 4) == 0;
}

std::string xml_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (char ch : value) {
        switch (ch) {
            case '&':
                escaped += "&amp;";
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '"':
                escaped += "&quot;";
                break;
            case '\'':
                escaped += "&apos;";
                break;
            default:
                escaped += ch;
                break;
        }
    }

    return escaped;
}

#ifdef SD_USE_WEBP
uint8_t* decode_webp_image_to_buffer(const uint8_t* data,
                                     size_t size,
                                     int& width,
                                     int& height,
                                     int expected_channel,
                                     int& source_channel_count) {
    WebPBitstreamFeatures features;
    if (WebPGetFeatures(data, size, &features) != VP8_STATUS_OK) {
        return nullptr;
    }

    width                = features.width;
    height               = features.height;
    source_channel_count = features.has_alpha ? 4 : 3;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

    if (expected_channel == 1) {
        int decoded_width  = width;
        int decoded_height = height;
        WebPBufferPtr decoded(features.has_alpha
                                  ? WebPDecodeRGBA(data, size, &decoded_width, &decoded_height)
                                  : WebPDecodeRGB(data, size, &decoded_width, &decoded_height));
        if (decoded == nullptr) {
            return nullptr;
        }

        FreeUniquePtr<uint8_t> grayscale((uint8_t*)malloc(pixel_count));
        if (grayscale == nullptr) {
            return nullptr;
        }

        const int decoded_channels = features.has_alpha ? 4 : 3;
        for (size_t i = 0; i < pixel_count; ++i) {
            const uint8_t* src = decoded.get() + i * decoded_channels;
            grayscale.get()[i] = static_cast<uint8_t>((77 * src[0] + 150 * src[1] + 29 * src[2] + 128) >> 8);
        }

        return grayscale.release();
    }

    if (expected_channel != 3 && expected_channel != 4) {
        return nullptr;
    }

    int decoded_width  = width;
    int decoded_height = height;
    WebPBufferPtr decoded((expected_channel == 4)
                              ? WebPDecodeRGBA(data, size, &decoded_width, &decoded_height)
                              : WebPDecodeRGB(data, size, &decoded_width, &decoded_height));
    if (decoded == nullptr) {
        return nullptr;
    }

    const size_t out_size = pixel_count * static_cast<size_t>(expected_channel);
    FreeUniquePtr<uint8_t> output((uint8_t*)malloc(out_size));
    if (output == nullptr) {
        return nullptr;
    }

    memcpy(output.get(), decoded.get(), out_size);
    return output.release();
}

std::string build_webp_xmp_packet(const std::string& parameters) {
    if (parameters.empty()) {
        return "";
    }

    const std::string escaped_parameters = xml_escape(parameters);
    return "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
           "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n"
           "  <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
           "    <rdf:Description xmlns:sdcpp=\"https://github.com/leejet/stable-diffusion.cpp/ns/1.0/\">\n"
           "      <sdcpp:parameters>" +
           escaped_parameters +
           "</sdcpp:parameters>\n"
           "    </rdf:Description>\n"
           "  </rdf:RDF>\n"
           "</x:xmpmeta>\n"
           "<?xpacket end=\"w\"?>";
}

bool encode_webp_image_to_vector(const uint8_t* image,
                                 int width,
                                 int height,
                                 int channels,
                                 const std::string& parameters,
                                 int quality,
                                 std::vector<uint8_t>& out) {
    if (image == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    std::vector<uint8_t> rgb_image;
    const uint8_t* input_image = image;
    int input_channels         = channels;

    if (channels == 1) {
        rgb_image.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_image[i * 3 + 0] = image[i];
            rgb_image[i * 3 + 1] = image[i];
            rgb_image[i * 3 + 2] = image[i];
        }
        input_image    = rgb_image.data();
        input_channels = 3;
    }

    if (input_channels != 3 && input_channels != 4) {
        return false;
    }

    uint8_t* encoded_raw = nullptr;
    size_t encoded_size  = (input_channels == 4)
                               ? WebPEncodeRGBA(input_image, width, height, width * input_channels, static_cast<float>(quality), &encoded_raw)
                               : WebPEncodeRGB(input_image, width, height, width * input_channels, static_cast<float>(quality), &encoded_raw);
    WebPBufferPtr encoded(encoded_raw);
    if (encoded == nullptr || encoded_size == 0) {
        return false;
    }

    out.assign(encoded.get(), encoded.get() + encoded_size);

    if (parameters.empty()) {
        return true;
    }

    WebPData image_data;
    WebPDataInit(&image_data);
    WebPDataGuard assembled_data;

    image_data.bytes = out.data();
    image_data.size  = out.size();

    WebPMuxPtr mux(WebPMuxNew());
    if (mux == nullptr) {
        return false;
    }

    const std::string xmp_packet = build_webp_xmp_packet(parameters);
    WebPData xmp_data;
    WebPDataInit(&xmp_data);
    xmp_data.bytes = reinterpret_cast<const uint8_t*>(xmp_packet.data());
    xmp_data.size  = xmp_packet.size();

    const bool ok = WebPMuxSetImage(mux.get(), &image_data, 1) == WEBP_MUX_OK &&
                    WebPMuxSetChunk(mux.get(), "XMP ", &xmp_data, 1) == WEBP_MUX_OK &&
                    WebPMuxAssemble(mux.get(), &assembled_data.data) == WEBP_MUX_OK;

    if (ok) {
        out.assign(assembled_data.data.bytes, assembled_data.data.bytes + assembled_data.data.size);
    }

    return ok;
}

#ifdef SD_USE_WEBM
bool extract_vp8_frame_from_webp(const std::vector<uint8_t>& webp_data, std::vector<uint8_t>& vp8_frame) {
    if (!is_webp_signature(webp_data.data(), webp_data.size())) {
        return false;
    }

    size_t offset = 12;
    while (offset + 8 <= webp_data.size()) {
        const uint8_t* chunk     = webp_data.data() + offset;
        const uint32_t chunk_len = read_u32_le_bytes(chunk + 4);
        const size_t chunk_start = offset + 8;
        const size_t padded_len  = static_cast<size_t>(chunk_len) + (chunk_len & 1u);

        if (chunk_start + chunk_len > webp_data.size()) {
            return false;
        }

        if (memcmp(chunk, "VP8 ", 4) == 0) {
            vp8_frame.assign(webp_data.data() + chunk_start,
                             webp_data.data() + chunk_start + chunk_len);
            return !vp8_frame.empty();
        }

        offset = chunk_start + padded_len;
    }

    return false;
}

bool encode_sd_image_to_vp8_frame(const sd_image_t& image, int quality, std::vector<uint8_t>& vp8_frame) {
    if (image.data == nullptr || image.width == 0 || image.height == 0) {
        return false;
    }

    const int width         = static_cast<int>(image.width);
    const int height        = static_cast<int>(image.height);
    const int input_channel = static_cast<int>(image.channel);
    if (input_channel != 1 && input_channel != 3 && input_channel != 4) {
        return false;
    }

    std::vector<uint8_t> rgb_buffer;
    const uint8_t* rgb_data = image.data;
    if (input_channel == 1) {
        rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_buffer[i * 3 + 0] = image.data[i];
            rgb_buffer[i * 3 + 1] = image.data[i];
            rgb_buffer[i * 3 + 2] = image.data[i];
        }
        rgb_data = rgb_buffer.data();
    } else if (input_channel == 4) {
        rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb_buffer[i * 3 + 0] = image.data[i * 4 + 0];
            rgb_buffer[i * 3 + 1] = image.data[i * 4 + 1];
            rgb_buffer[i * 3 + 2] = image.data[i * 4 + 2];
        }
        rgb_data = rgb_buffer.data();
    }

    std::vector<uint8_t> encoded_webp;
    if (!encode_webp_image_to_vector(rgb_data, width, height, 3, "", quality, encoded_webp)) {
        return false;
    }

    return extract_vp8_frame_from_webp(encoded_webp, vp8_frame);
}
#endif
#endif

uint8_t* load_image_common(bool from_memory,
                           const char* image_path_or_bytes,
                           int len,
                           int& width,
                           int& height,
                           int expected_width,
                           int expected_height,
                           int expected_channel) {
    const char* image_path;
    FreeUniquePtr<uint8_t> image_buffer;
    int source_channel_count = 0;

#ifdef SD_USE_WEBP
    if (from_memory) {
        image_path = "memory";
        if (len > 0 && is_webp_signature(reinterpret_cast<const uint8_t*>(image_path_or_bytes), static_cast<size_t>(len))) {
            image_buffer.reset(decode_webp_image_to_buffer(reinterpret_cast<const uint8_t*>(image_path_or_bytes),
                                                           static_cast<size_t>(len),
                                                           width,
                                                           height,
                                                           expected_channel,
                                                           source_channel_count));
        }
    } else {
        image_path = image_path_or_bytes;
        if (encoded_image_format_from_path(image_path_or_bytes) == EncodedImageFormat::WEBP) {
            std::vector<uint8_t> file_bytes;
            if (!read_binary_file_bytes(image_path_or_bytes, file_bytes)) {
                LOG_ERROR("load image from '%s' failed", image_path_or_bytes);
                return nullptr;
            }
            if (!is_webp_signature(file_bytes.data(), file_bytes.size())) {
                LOG_ERROR("load image from '%s' failed", image_path_or_bytes);
                return nullptr;
            }
            image_buffer.reset(decode_webp_image_to_buffer(file_bytes.data(),
                                                           file_bytes.size(),
                                                           width,
                                                           height,
                                                           expected_channel,
                                                           source_channel_count));
        }
    }
#endif

    if (from_memory) {
        image_path = "memory";
        if (image_buffer == nullptr) {
            int c = 0;
            image_buffer.reset((uint8_t*)stbi_load_from_memory((const stbi_uc*)image_path_or_bytes, len, &width, &height, &c, expected_channel));
            source_channel_count = c;
        }
    } else {
        image_path = image_path_or_bytes;
        if (image_buffer == nullptr) {
            int c = 0;
            image_buffer.reset((uint8_t*)stbi_load(image_path_or_bytes, &width, &height, &c, expected_channel));
            source_channel_count = c;
        }
    }
    if (image_buffer == nullptr) {
        LOG_ERROR("load image from '%s' failed", image_path);
        return nullptr;
    }
    if (source_channel_count < expected_channel) {
        fprintf(stderr,
                "the number of channels for the input image must be >= %d,"
                "but got %d channels, image_path = %s",
                expected_channel,
                source_channel_count,
                image_path);
        return nullptr;
    }
    if (width <= 0) {
        LOG_ERROR("error: the width of image must be greater than 0, image_path = %s", image_path);
        return nullptr;
    }
    if (height <= 0) {
        LOG_ERROR("error: the height of image must be greater than 0, image_path = %s", image_path);
        return nullptr;
    }

    if ((expected_width > 0 && expected_height > 0) && (height != expected_height || width != expected_width)) {
        float dst_aspect = (float)expected_width / (float)expected_height;
        float src_aspect = (float)width / (float)height;

        int crop_x = 0, crop_y = 0;
        int crop_w = width, crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = (int)(height * dst_aspect);
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = (int)(width / dst_aspect);
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0) {
            LOG_INFO("crop input image from %dx%d to %dx%d, image_path = %s", width, height, crop_w, crop_h, image_path);
            FreeUniquePtr<uint8_t> cropped_image_buffer((uint8_t*)malloc(crop_w * crop_h * expected_channel));
            if (cropped_image_buffer == nullptr) {
                LOG_ERROR("error: allocate memory for crop\n");
                return nullptr;
            }
            for (int row = 0; row < crop_h; row++) {
                uint8_t* src = image_buffer.get() + ((crop_y + row) * width + crop_x) * expected_channel;
                uint8_t* dst = cropped_image_buffer.get() + (row * crop_w) * expected_channel;
                memcpy(dst, src, crop_w * expected_channel);
            }

            width        = crop_w;
            height       = crop_h;
            image_buffer = std::move(cropped_image_buffer);
        }

        LOG_INFO("resize input image from %dx%d to %dx%d", width, height, expected_width, expected_height);
        FreeUniquePtr<uint8_t> resized_image_buffer((uint8_t*)malloc(expected_height * expected_width * expected_channel));
        if (resized_image_buffer == nullptr) {
            LOG_ERROR("error: allocate memory for resize input image\n");
            return nullptr;
        }
        stbir_resize(image_buffer.get(), width, height, 0,
                     resized_image_buffer.get(), expected_width, expected_height, 0, STBIR_TYPE_UINT8,
                     expected_channel, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        width        = expected_width;
        height       = expected_height;
        image_buffer = std::move(resized_image_buffer);
    }
    return image_buffer.release();
}

typedef struct {
    uint32_t offset;
    uint32_t size;
} avi_index_entry;

void write_u32_le(FILE* f, uint32_t val) {
    fwrite(&val, 4, 1, f);
}

void write_u16_le(FILE* f, uint16_t val) {
    fwrite(&val, 2, 1, f);
}

void write_u32_le(std::vector<uint8_t>& data, uint32_t val) {
    data.push_back(static_cast<uint8_t>(val & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

void write_u16_le(std::vector<uint8_t>& data, uint16_t val) {
    data.push_back(static_cast<uint8_t>(val & 0xFF));
    data.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
}

void patch_u32_le(std::vector<uint8_t>& data, size_t offset, uint32_t val) {
    if (offset + 4 > data.size()) {
        return;
    }
    data[offset + 0] = static_cast<uint8_t>(val & 0xFF);
    data[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    data[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    data[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

void write_fourcc(std::vector<uint8_t>& data, const char* fourcc) {
    data.insert(data.end(), fourcc, fourcc + 4);
}

EncodedImageFormat encoded_image_format_from_path(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".jpg" || ext == ".jpeg" || ext == ".jpe") {
        return EncodedImageFormat::JPEG;
    }
    if (ext == ".png") {
        return EncodedImageFormat::PNG;
    }
    if (ext == ".webp") {
        return EncodedImageFormat::WEBP;
    }
    return EncodedImageFormat::UNKNOWN;
}

std::vector<uint8_t> encode_image_to_vector(EncodedImageFormat format,
                                            const uint8_t* image,
                                            int width,
                                            int height,
                                            int channels,
                                            const std::string& parameters,
                                            int quality) {
    std::vector<uint8_t> buffer;

    auto write_func = [&buffer](void* context, void* data, int size) {
        (void)context;
        uint8_t* src = reinterpret_cast<uint8_t*>(data);
        buffer.insert(buffer.end(), src, src + size);
    };

    struct ContextWrapper {
        decltype(write_func)& func;
    } ctx{write_func};

    auto c_func = [](void* context, void* data, int size) {
        auto* wrapper = reinterpret_cast<ContextWrapper*>(context);
        wrapper->func(context, data, size);
    };

    int result = 0;
    switch (format) {
        case EncodedImageFormat::JPEG:
            result = stbi_write_jpg_to_func(c_func, &ctx, width, height, channels, image, quality);
            break;
        case EncodedImageFormat::PNG:
            result = stbi_ext_write_png_to_func(c_func, &ctx, width, height, channels, image, width * channels, parameters.empty() ? nullptr : parameters.c_str());
            break;
        case EncodedImageFormat::WEBP:
#ifdef SD_USE_WEBP
            if (!encode_webp_image_to_vector(image, width, height, channels, parameters, quality, buffer)) {
                buffer.clear();
            }
            result = buffer.empty() ? 0 : 1;
            break;
#else
            result = 0;
            break;
#endif
        default:
            result = 0;
            break;
    }

    if (!result) {
        buffer.clear();
    }
    return buffer;
}

bool write_image_to_file(const std::string& path,
                         const uint8_t* image,
                         int width,
                         int height,
                         int channels,
                         const std::string& parameters,
                         int quality) {
    const EncodedImageFormat format = encoded_image_format_from_path(path);

    switch (format) {
        case EncodedImageFormat::JPEG:
            return stbi_write_jpg(path.c_str(), width, height, channels, image, quality, parameters.empty() ? nullptr : parameters.c_str()) != 0;
        case EncodedImageFormat::PNG:
            return stbi_write_png(path.c_str(), width, height, channels, image, 0, parameters.empty() ? nullptr : parameters.c_str()) != 0;
        case EncodedImageFormat::WEBP: {
            const std::vector<uint8_t> encoded = encode_image_to_vector(format, image, width, height, channels, parameters, quality);
            return !encoded.empty() && write_binary_file_bytes(path, encoded);
        }
        default:
            return false;
    }
}

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width,
                              int expected_height,
                              int expected_channel) {
    return load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
}

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width,
                             int expected_height,
                             int expected_channel) {
    int width;
    int height;
    image->data = load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
    if (image->data == nullptr) {
        return false;
    }
    image->width   = width;
    image->height  = height;
    image->channel = expected_channel;
    return true;
}

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width,
                                int expected_height,
                                int expected_channel) {
    return load_image_common(true, image_bytes, len, width, height, expected_width, expected_height, expected_channel);
}

std::vector<uint8_t> create_mjpg_avi_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }

    uint32_t width    = images[0].width;
    uint32_t height   = images[0].height;
    uint32_t channels = images[0].channel;
    if (channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %u\n", channels);
        return {};
    }

    // stb_image_write changes JPEG sampling behavior above quality 90.
    // MJPG AVI playback is more compatible when we keep the encoder on the
    // <= 90 path.
    const int mjpg_quality = std::clamp(quality, 1, 90);

    std::vector<uint8_t> avi_data;
    avi_data.reserve(static_cast<size_t>(num_images) * 1024);

    write_fourcc(avi_data, "RIFF");
    const size_t riff_size_pos = avi_data.size();
    write_u32_le(avi_data, 0);
    write_fourcc(avi_data, "AVI ");

    write_fourcc(avi_data, "LIST");
    write_u32_le(avi_data, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi_data, "hdrl");

    write_fourcc(avi_data, "avih");
    write_u32_le(avi_data, 56);
    write_u32_le(avi_data, 1000000 / fps);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0x110);
    write_u32_le(avi_data, num_images);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 1);
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, width);
    write_u32_le(avi_data, height);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);

    write_fourcc(avi_data, "LIST");
    write_u32_le(avi_data, 4 + 8 + 56 + 8 + 40);
    write_fourcc(avi_data, "strl");

    write_fourcc(avi_data, "strh");
    write_u32_le(avi_data, 56);
    write_fourcc(avi_data, "vids");
    write_fourcc(avi_data, "MJPG");
    write_u32_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 1);
    write_u32_le(avi_data, fps);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, num_images);
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, static_cast<uint32_t>(-1));
    write_u32_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);
    write_u16_le(avi_data, 0);

    write_fourcc(avi_data, "strf");
    write_u32_le(avi_data, 40);
    write_u32_le(avi_data, 40);
    write_u32_le(avi_data, width);
    write_u32_le(avi_data, height);
    write_u16_le(avi_data, 1);
    write_u16_le(avi_data, 24);
    write_fourcc(avi_data, "MJPG");
    write_u32_le(avi_data, width * height * 3);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);
    write_u32_le(avi_data, 0);

    write_fourcc(avi_data, "LIST");
    const size_t movi_size_pos = avi_data.size();
    write_u32_le(avi_data, 0);
    write_fourcc(avi_data, "movi");

    std::vector<avi_index_entry> index(static_cast<size_t>(num_images));
    std::vector<uint8_t> jpeg_data;

    for (int i = 0; i < num_images; i++) {
        jpeg_data.clear();

        auto write_to_buf = [](void* context, void* data, int size) {
            auto* buffer       = reinterpret_cast<std::vector<uint8_t>*>(context);
            const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
            buffer->insert(buffer->end(), src, src + size);
        };

        if (!stbi_write_jpg_to_func(write_to_buf, &jpeg_data, images[i].width, images[i].height, channels, images[i].data, mjpg_quality)) {
            fprintf(stderr, "Error: Failed to encode JPEG frame.\n");
            return {};
        }

        index[i].offset = static_cast<uint32_t>(avi_data.size());
        write_fourcc(avi_data, "00dc");
        write_u32_le(avi_data, static_cast<uint32_t>(jpeg_data.size()));
        index[i].size = (uint32_t)jpeg_data.size();
        avi_data.insert(avi_data.end(), jpeg_data.begin(), jpeg_data.end());

        if (jpeg_data.size() % 2) {
            avi_data.push_back(0);
        }
    }

    const size_t movi_size = avi_data.size() - movi_size_pos - 4;
    patch_u32_le(avi_data, movi_size_pos, static_cast<uint32_t>(movi_size));

    write_fourcc(avi_data, "idx1");
    write_u32_le(avi_data, num_images * 16);
    for (int i = 0; i < num_images; i++) {
        write_fourcc(avi_data, "00dc");
        write_u32_le(avi_data, 0x10);
        write_u32_le(avi_data, index[i].offset);
        write_u32_le(avi_data, index[i].size);
    }

    const size_t file_size = avi_data.size() - riff_size_pos - 4;
    patch_u32_le(avi_data, riff_size_pos, static_cast<uint32_t>(file_size));

    return avi_data;
}

int create_mjpg_avi_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality) {
    std::vector<uint8_t> avi_data = create_mjpg_avi_from_sd_images_to_vector(images, num_images, fps, quality);
    if (avi_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, avi_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}

#ifdef SD_USE_WEBP
std::vector<uint8_t> create_animated_webp_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }
    if (fps <= 0) {
        fprintf(stderr, "Error: FPS must be positive.\n");
        return {};
    }

    const int width    = static_cast<int>(images[0].width);
    const int height   = static_cast<int>(images[0].height);
    const int channels = static_cast<int>(images[0].channel);
    if (channels != 1 && channels != 3 && channels != 4) {
        fprintf(stderr, "Error: Unsupported channel count: %d\n", channels);
        return {};
    }

    WebPAnimEncoderOptions anim_options;
    WebPConfig config;
    if (!WebPAnimEncoderOptionsInit(&anim_options) || !WebPConfigInit(&config)) {
        fprintf(stderr, "Error: Failed to initialize WebP animation encoder.\n");
        return {};
    }

    config.quality      = static_cast<float>(quality);
    config.method       = 4;
    config.thread_level = 1;
    if (channels == 4) {
        config.exact = 1;
    }
    if (!WebPValidateConfig(&config)) {
        fprintf(stderr, "Error: Invalid WebP encoder configuration.\n");
        return {};
    }

    WebPAnimEncoderPtr enc(WebPAnimEncoderNew(width, height, &anim_options));
    if (enc == nullptr) {
        fprintf(stderr, "Error: Could not create WebPAnimEncoder object.\n");
        return {};
    }

    const int frame_duration_ms = std::max(1, static_cast<int>(std::lround(1000.0 / static_cast<double>(fps))));
    int timestamp_ms            = 0;

    for (int i = 0; i < num_images; ++i) {
        const sd_image_t& image = images[i];
        if (static_cast<int>(image.width) != width || static_cast<int>(image.height) != height) {
            fprintf(stderr, "Error: Frame dimensions do not match.\n");
            return {};
        }

        WebPPictureGuard picture;
        if (!picture.initialized) {
            fprintf(stderr, "Error: Failed to initialize WebPPicture.\n");
            return {};
        }
        picture.picture.use_argb = 1;
        picture.picture.width    = width;
        picture.picture.height   = height;

        bool picture_ok = false;
        std::vector<uint8_t> rgb_buffer;
        if (image.channel == 1) {
            rgb_buffer.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
            for (int p = 0; p < width * height; ++p) {
                rgb_buffer[p * 3 + 0] = image.data[p];
                rgb_buffer[p * 3 + 1] = image.data[p];
                rgb_buffer[p * 3 + 2] = image.data[p];
            }
            picture_ok = WebPPictureImportRGB(&picture.picture, rgb_buffer.data(), width * 3) != 0;
        } else if (image.channel == 4) {
            picture_ok = WebPPictureImportRGBA(&picture.picture, image.data, width * 4) != 0;
        } else {
            picture_ok = WebPPictureImportRGB(&picture.picture, image.data, width * 3) != 0;
        }

        if (!picture_ok) {
            fprintf(stderr, "Error: Failed to import frame into WebPPicture.\n");
            return {};
        }

        if (!WebPAnimEncoderAdd(enc.get(), &picture.picture, timestamp_ms, &config)) {
            fprintf(stderr, "Error: Failed to add frame to animated WebP: %s\n", WebPAnimEncoderGetError(enc.get()));
            return {};
        }

        timestamp_ms += frame_duration_ms;
    }

    if (!WebPAnimEncoderAdd(enc.get(), nullptr, timestamp_ms, nullptr)) {
        fprintf(stderr, "Error: Failed to finalize animated WebP frames: %s\n", WebPAnimEncoderGetError(enc.get()));
        return {};
    }

    WebPDataGuard webp_data;
    if (!WebPAnimEncoderAssemble(enc.get(), &webp_data.data)) {
        fprintf(stderr, "Error: Failed to assemble animated WebP: %s\n", WebPAnimEncoderGetError(enc.get()));
        return {};
    }

    return std::vector<uint8_t>(webp_data.data.bytes, webp_data.data.bytes + webp_data.data.size);
}

int create_animated_webp_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality) {
    std::vector<uint8_t> webp_data = create_animated_webp_from_sd_images_to_vector(images, num_images, fps, quality);
    if (webp_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, webp_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}
#endif

#ifdef SD_USE_WEBM
std::vector<uint8_t> create_webm_from_sd_images_to_vector(sd_image_t* images, int num_images, int fps, int quality) {
    if (num_images == 0) {
        fprintf(stderr, "Error: Image array is empty.\n");
        return {};
    }
    if (fps <= 0) {
        fprintf(stderr, "Error: FPS must be positive.\n");
        return {};
    }

    const int width  = static_cast<int>(images[0].width);
    const int height = static_cast<int>(images[0].height);
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid frame dimensions.\n");
        return {};
    }

    MemoryMkvWriter writer;

    const int ret = [&]() -> int {
        mkvmuxer::Segment segment;
        if (!segment.Init(&writer)) {
            fprintf(stderr, "Error: Failed to initialize WebM muxer.\n");
            return -1;
        }

        segment.set_mode(mkvmuxer::Segment::kFile);
        segment.OutputCues(true);

        const uint64_t track_number = segment.AddVideoTrack(width, height, 0);
        if (track_number == 0) {
            fprintf(stderr, "Error: Failed to add VP8 video track.\n");
            return -1;
        }
        if (!segment.CuesTrack(track_number)) {
            fprintf(stderr, "Error: Failed to set WebM cues track.\n");
            return -1;
        }

        mkvmuxer::VideoTrack* video_track = static_cast<mkvmuxer::VideoTrack*>(segment.GetTrackByNumber(track_number));
        if (video_track != nullptr) {
            video_track->set_display_width(static_cast<uint64_t>(width));
            video_track->set_display_height(static_cast<uint64_t>(height));
            video_track->set_frame_rate(static_cast<double>(fps));
        }
        segment.GetSegmentInfo()->set_writing_app("stable-diffusion.cpp");
        segment.GetSegmentInfo()->set_muxing_app("stable-diffusion.cpp");

        const uint64_t frame_duration_ns = std::max<uint64_t>(
            1, static_cast<uint64_t>(std::llround(1000000000.0 / static_cast<double>(fps))));
        uint64_t timestamp_ns = 0;

        for (int i = 0; i < num_images; ++i) {
            const sd_image_t& image = images[i];
            if (static_cast<int>(image.width) != width || static_cast<int>(image.height) != height) {
                fprintf(stderr, "Error: Frame dimensions do not match.\n");
                return -1;
            }

            std::vector<uint8_t> vp8_frame;
            if (!encode_sd_image_to_vp8_frame(image, quality, vp8_frame)) {
                fprintf(stderr, "Error: Failed to encode frame %d as VP8.\n", i);
                return -1;
            }

            if (!segment.AddFrame(vp8_frame.data(),
                                  static_cast<uint64_t>(vp8_frame.size()),
                                  track_number,
                                  timestamp_ns,
                                  true)) {
                fprintf(stderr, "Error: Failed to mux frame %d into WebM.\n", i);
                return -1;
            }

            timestamp_ns += frame_duration_ns;
        }

        if (!segment.Finalize()) {
            fprintf(stderr, "Error: Failed to finalize WebM output.\n");
            return -1;
        }
        return 0;
    }();
    if (ret != 0) {
        return {};
    }
    return writer.data();
}

int create_webm_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality) {
    std::vector<uint8_t> webm_data = create_webm_from_sd_images_to_vector(images, num_images, fps, quality);
    if (webm_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, webm_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}
#endif

std::vector<uint8_t> create_video_from_sd_images_to_vector(const std::string& output_format,
                                                           sd_image_t* images,
                                                           int num_images,
                                                           int fps,
                                                           int quality) {
    std::string format = output_format;
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char c) { return static_cast<char>(tolower(c)); });
    if (!format.empty() && format[0] == '.') {
        format.erase(format.begin());
    }

#ifdef SD_USE_WEBM
    if (format == "webm") {
        return create_webm_from_sd_images_to_vector(images, num_images, fps, quality);
    }
#endif

#ifdef SD_USE_WEBP
    if (format == "webp") {
        return create_animated_webp_from_sd_images_to_vector(images, num_images, fps, quality);
    }
#endif

    return create_mjpg_avi_from_sd_images_to_vector(images, num_images, fps, quality);
}

int create_video_from_sd_images(const char* filename, sd_image_t* images, int num_images, int fps, int quality) {
    std::string path                = filename ? filename : "";
    auto pos                        = path.find_last_of('.');
    std::string ext                 = pos == std::string::npos ? "" : path.substr(pos);
    std::vector<uint8_t> video_data = create_video_from_sd_images_to_vector(ext, images, num_images, fps, quality);
    if (video_data.empty()) {
        return -1;
    }
    if (!write_binary_file_bytes(filename, video_data)) {
        perror("Error opening file for writing");
        return -1;
    }
    return 0;
}

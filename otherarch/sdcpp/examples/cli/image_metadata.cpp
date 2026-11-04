#include "image_metadata.h"

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <json.hpp>

using json = nlohmann::json;

extern "C" int mz_uncompress(unsigned char* pDest,
                             unsigned long* pDest_len,
                             const unsigned char* pSource,
                             unsigned long source_len);

namespace {

    constexpr int MZ_OK        = 0;
    constexpr int MZ_BUF_ERROR = -5;

    uint16_t read_u16_be(const std::vector<uint8_t>& data, size_t offset) {
        return (static_cast<uint16_t>(data[offset]) << 8) |
               static_cast<uint16_t>(data[offset + 1]);
    }

    uint32_t read_u32_be(const std::vector<uint8_t>& data, size_t offset) {
        return (static_cast<uint32_t>(data[offset]) << 24) |
               (static_cast<uint32_t>(data[offset + 1]) << 16) |
               (static_cast<uint32_t>(data[offset + 2]) << 8) |
               static_cast<uint32_t>(data[offset + 3]);
    }

    uint32_t read_u32_le(const std::vector<uint8_t>& data, size_t offset) {
        return static_cast<uint32_t>(data[offset]) |
               (static_cast<uint32_t>(data[offset + 1]) << 8) |
               (static_cast<uint32_t>(data[offset + 2]) << 16) |
               (static_cast<uint32_t>(data[offset + 3]) << 24);
    }

    uint16_t read_u16_tiff(const std::vector<uint8_t>& data, size_t offset, bool little_endian) {
        if (little_endian) {
            return static_cast<uint16_t>(data[offset]) |
                   (static_cast<uint16_t>(data[offset + 1]) << 8);
        }
        return read_u16_be(data, offset);
    }

    uint32_t read_u32_tiff(const std::vector<uint8_t>& data, size_t offset, bool little_endian) {
        if (little_endian) {
            return static_cast<uint32_t>(data[offset]) |
                   (static_cast<uint32_t>(data[offset + 1]) << 8) |
                   (static_cast<uint32_t>(data[offset + 2]) << 16) |
                   (static_cast<uint32_t>(data[offset + 3]) << 24);
        }
        return read_u32_be(data, offset);
    }

    int32_t read_i32_tiff(const std::vector<uint8_t>& data, size_t offset, bool little_endian) {
        return static_cast<int32_t>(read_u32_tiff(data, offset, little_endian));
    }

    std::string bytes_to_string(const uint8_t* begin, const uint8_t* end) {
        return std::string(reinterpret_cast<const char*>(begin),
                           reinterpret_cast<const char*>(end));
    }

    std::string trim_trailing_nuls(std::string value) {
        while (!value.empty() && value.back() == '\0') {
            value.pop_back();
        }
        return value;
    }

    std::string hex_preview(const uint8_t* data, size_t size, size_t limit = 64) {
        std::ostringstream oss;
        const size_t count = std::min(size, limit);
        for (size_t i = 0; i < count; ++i) {
            if (i != 0) {
                oss << ' ';
            }
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(data[i]);
        }
        if (size > limit) {
            oss << " ...";
        }
        return oss.str();
    }

    std::string marker_name(uint8_t marker) {
        if (marker >= 0xE0 && marker <= 0xEF) {
            return "APP" + std::to_string(marker - 0xE0);
        }
        if (marker == 0xFE) {
            return "COM";
        }
        if (marker == 0xDA) {
            return "SOS";
        }
        if (marker == 0xD8) {
            return "SOI";
        }
        if (marker == 0xD9) {
            return "EOI";
        }
        if (marker >= 0xD0 && marker <= 0xD7) {
            return "RST" + std::to_string(marker - 0xD0);
        }
        std::ostringstream oss;
        oss << "0x" << std::hex << std::uppercase << static_cast<int>(marker);
        return oss.str();
    }

    std::string exif_type_name(uint16_t type) {
        switch (type) {
            case 1:
                return "BYTE";
            case 2:
                return "ASCII";
            case 3:
                return "SHORT";
            case 4:
                return "LONG";
            case 5:
                return "RATIONAL";
            case 7:
                return "UNDEFINED";
            case 9:
                return "SLONG";
            case 10:
                return "SRATIONAL";
            default:
                return "UNKNOWN";
        }
    }

    size_t exif_type_size(uint16_t type) {
        switch (type) {
            case 1:
            case 2:
            case 7:
                return 1;
            case 3:
                return 2;
            case 4:
            case 9:
                return 4;
            case 5:
            case 10:
                return 8;
            default:
                return 0;
        }
    }

    const std::map<uint16_t, const char*>& tiff_tag_names() {
        static const std::map<uint16_t, const char*> names = {
            {0x0100, "ImageWidth"},
            {0x0101, "ImageLength"},
            {0x0102, "BitsPerSample"},
            {0x0103, "Compression"},
            {0x0106, "PhotometricInterpretation"},
            {0x010E, "ImageDescription"},
            {0x010F, "Make"},
            {0x0110, "Model"},
            {0x0111, "StripOffsets"},
            {0x0112, "Orientation"},
            {0x0115, "SamplesPerPixel"},
            {0x011A, "XResolution"},
            {0x011B, "YResolution"},
            {0x0128, "ResolutionUnit"},
            {0x0131, "Software"},
            {0x0132, "DateTime"},
            {0x013B, "Artist"},
            {0x8298, "Copyright"},
            {0x8769, "ExifIFDPointer"},
            {0x8825, "GPSInfoIFDPointer"},
        };
        return names;
    }

    const std::map<uint16_t, const char*>& exif_tag_names() {
        static const std::map<uint16_t, const char*> names = {
            {0x829A, "ExposureTime"},
            {0x829D, "FNumber"},
            {0x8827, "ISOSpeedRatings"},
            {0x9000, "ExifVersion"},
            {0x9003, "DateTimeOriginal"},
            {0x9004, "DateTimeDigitized"},
            {0x9201, "ShutterSpeedValue"},
            {0x9202, "ApertureValue"},
            {0x9204, "ExposureBiasValue"},
            {0x9209, "Flash"},
            {0x920A, "FocalLength"},
            {0x927C, "MakerNote"},
            {0x9286, "UserComment"},
            {0xA001, "ColorSpace"},
            {0xA002, "PixelXDimension"},
            {0xA003, "PixelYDimension"},
            {0xA005, "InteroperabilityIFDPointer"},
            {0xA402, "ExposureMode"},
            {0xA403, "WhiteBalance"},
            {0xA404, "DigitalZoomRatio"},
            {0xA405, "FocalLengthIn35mmFilm"},
            {0xA430, "CameraOwnerName"},
            {0xA431, "BodySerialNumber"},
            {0xA432, "LensSpecification"},
            {0xA433, "LensMake"},
            {0xA434, "LensModel"},
            {0xA435, "LensSerialNumber"},
        };
        return names;
    }

    const std::map<uint16_t, const char*>& gps_tag_names() {
        static const std::map<uint16_t, const char*> names = {
            {0x0000, "GPSVersionID"},
            {0x0001, "GPSLatitudeRef"},
            {0x0002, "GPSLatitude"},
            {0x0003, "GPSLongitudeRef"},
            {0x0004, "GPSLongitude"},
            {0x0005, "GPSAltitudeRef"},
            {0x0006, "GPSAltitude"},
            {0x0007, "GPSTimeStamp"},
            {0x000D, "GPSSpeed"},
            {0x0011, "GPSImgDirection"},
            {0x001D, "GPSDateStamp"},
        };
        return names;
    }

    const std::map<uint16_t, const char*>& interoperability_tag_names() {
        static const std::map<uint16_t, const char*> names = {
            {0x0001, "InteroperabilityIndex"},
            {0x0002, "InteroperabilityVersion"},
            {0x1000, "RelatedImageFileFormat"},
            {0x1001, "RelatedImageWidth"},
            {0x1002, "RelatedImageLength"},
        };
        return names;
    }

    const std::map<uint16_t, const char*>& tag_names_for_ifd(const std::string& ifd_name) {
        if (ifd_name == "Exif") {
            return exif_tag_names();
        }
        if (ifd_name == "GPS") {
            return gps_tag_names();
        }
        if (ifd_name == "Interop") {
            return interoperability_tag_names();
        }
        return tiff_tag_names();
    }

    std::string tag_name_for(uint16_t tag, const std::string& ifd_name) {
        const auto& names = tag_names_for_ifd(ifd_name);
        auto it           = names.find(tag);
        if (it != names.end()) {
            return it->second;
        }

        std::ostringstream oss;
        oss << "0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << tag;
        return oss.str();
    }

    bool read_file(const std::string& path, std::vector<uint8_t>& data, std::string& error) {
        std::ifstream fin(std::filesystem::u8path(path), std::ios::binary);
        if (!fin) {
            error = "failed to open file: " + path;
            return false;
        }
        fin.seekg(0, std::ios::end);
        std::streampos size = fin.tellg();
        if (size < 0) {
            error = "failed to read file size: " + path;
            return false;
        }
        fin.seekg(0, std::ios::beg);

        data.resize(static_cast<size_t>(size));
        if (!data.empty()) {
            fin.read(reinterpret_cast<char*>(data.data()), size);
            if (!fin) {
                error = "failed to read file: " + path;
                return false;
            }
        }
        return true;
    }

    bool decompress_zlib(const uint8_t* data,
                         size_t size,
                         std::string& text,
                         std::string& error) {
        if (size == 0) {
            text.clear();
            return true;
        }

        size_t capacity = std::max<size_t>(256, size * 4);
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::vector<unsigned char> buffer(capacity);
            unsigned long dest_len = static_cast<unsigned long>(buffer.size());
            int status             = mz_uncompress(buffer.data(),
                                                   &dest_len,
                                                   reinterpret_cast<const unsigned char*>(data),
                                                   static_cast<unsigned long>(size));
            if (status == MZ_OK) {
                text.assign(reinterpret_cast<const char*>(buffer.data()), dest_len);
                return true;
            }
            if (status != MZ_BUF_ERROR) {
                std::ostringstream oss;
                oss << "zlib decompression failed with status " << status;
                error = oss.str();
                return false;
            }
            capacity *= 2;
        }

        error = "zlib decompression exceeded retry budget";
        return false;
    }

    void append_raw_preview(json& entry, const uint8_t* data, size_t size, bool include_raw) {
        if (include_raw) {
            entry["raw_hex_preview"] = hex_preview(data, size);
        }
    }

    json parse_ifd(const std::vector<uint8_t>& data,
                   size_t tiff_start,
                   uint32_t offset,
                   bool little_endian,
                   const std::string& ifd_name,
                   bool include_raw,
                   std::set<uint32_t>& visited,
                   std::string& warning);

    json parse_exif_tiff(const uint8_t* payload,
                         size_t size,
                         bool include_raw,
                         std::string& error);

    bool parse_png(const std::vector<uint8_t>& data,
                   bool include_raw,
                   json& result,
                   std::string& error);

    bool parse_jpeg(const std::vector<uint8_t>& data,
                    bool include_raw,
                    json& result,
                    std::string& error);

    bool parse_webp(const std::vector<uint8_t>& data,
                    bool include_raw,
                    json& result,
                    std::string& error);

    std::string abbreviate(const std::string& value, bool brief);

    void print_json_value(std::ostream& out,
                          const std::string& key,
                          const json& value,
                          int indent,
                          bool brief);

    void print_text_report(const std::string& path,
                           const json& report,
                           bool include_structural,
                           bool brief,
                           std::ostream& out);

    json filter_visible_entries(const json& report, bool include_structural);

    bool build_metadata_report(const std::string& image_path,
                               bool include_raw,
                               json& report,
                               std::string& error);

    json parse_exif_user_comment(const uint8_t* data, size_t size) {
        if (size >= 8 && memcmp(data, "ASCII\0\0\0", 8) == 0) {
            return trim_trailing_nuls(bytes_to_string(data + 8, data + size));
        }
        if (size < 8 || memcmp(data, "UNICODE\0", 8) != 0) {
            return json{{"raw_hex_preview", hex_preview(data, size)}};
        }
        size_t offset = 8;
        // WebUI's piexif writer uses big-endian UTF-16 without a BOM.
        bool little_endian = false;
        if (size >= 10 && ((data[8] == 0xfe && data[9] == 0xff) || (data[8] == 0xff && data[9] == 0xfe))) {
            little_endian = data[8] == 0xff;
            offset        = 10;
        }
        if ((size - offset) % 2 != 0) {
            return json{{"error", "truncated UTF-16 UserComment"}, {"raw_hex_preview", hex_preview(data, size)}};
        }
        std::u16string text;
        for (; offset < size; offset += 2) {
            text.push_back(static_cast<char16_t>(little_endian ? data[offset] | (uint16_t(data[offset + 1]) << 8)
                                                               : (uint16_t(data[offset]) << 8) | data[offset + 1]));
        }
        try {
            std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
            return trim_trailing_nuls(converter.to_bytes(text));
        } catch (const std::range_error&) {
            return json{{"error", "invalid UTF-16 UserComment"}, {"raw_hex_preview", hex_preview(data, size)}};
        }
    }

    json parse_exif_value(const std::vector<uint8_t>& data,
                          size_t value_offset,
                          uint16_t tag,
                          uint16_t type,
                          uint32_t count,
                          bool little_endian,
                          bool include_raw,
                          const uint8_t* raw_ptr,
                          size_t raw_size) {
        json value;
        switch (type) {
            case 1: {
                if (count == 1) {
                    value = data[value_offset];
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(data[value_offset + i]);
                    }
                    value = std::move(arr);
                }
                break;
            }
            case 2:
                value = trim_trailing_nuls(bytes_to_string(data.data() + value_offset,
                                                           data.data() + value_offset + count));
                break;
            case 3: {
                if (count == 1) {
                    value = read_u16_tiff(data, value_offset, little_endian);
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(read_u16_tiff(data, value_offset + i * 2, little_endian));
                    }
                    value = std::move(arr);
                }
                break;
            }
            case 4: {
                if (count == 1) {
                    value = read_u32_tiff(data, value_offset, little_endian);
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(read_u32_tiff(data, value_offset + i * 4, little_endian));
                    }
                    value = std::move(arr);
                }
                break;
            }
            case 5: {
                auto read_rational = [&](size_t off) {
                    uint32_t num = read_u32_tiff(data, off, little_endian);
                    uint32_t den = read_u32_tiff(data, off + 4, little_endian);
                    std::ostringstream oss;
                    oss << num << "/" << den;
                    if (den != 0) {
                        oss << " (" << std::fixed << std::setprecision(6)
                            << static_cast<double>(num) / static_cast<double>(den) << ")";
                    }
                    return oss.str();
                };

                if (count == 1) {
                    value = read_rational(value_offset);
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(read_rational(value_offset + i * 8));
                    }
                    value = std::move(arr);
                }
                break;
            }
            case 7:
                value = tag == 0x9286 ? parse_exif_user_comment(data.data() + value_offset, count)
                                      : json(bytes_to_string(data.data() + value_offset,
                                                             data.data() + value_offset + count));
                break;
            case 9: {
                if (count == 1) {
                    value = read_i32_tiff(data, value_offset, little_endian);
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(read_i32_tiff(data, value_offset + i * 4, little_endian));
                    }
                    value = std::move(arr);
                }
                break;
            }
            case 10: {
                auto read_srational = [&](size_t off) {
                    int32_t num = read_i32_tiff(data, off, little_endian);
                    int32_t den = read_i32_tiff(data, off + 4, little_endian);
                    std::ostringstream oss;
                    oss << num << "/" << den;
                    if (den != 0) {
                        oss << " (" << std::fixed << std::setprecision(6)
                            << static_cast<double>(num) / static_cast<double>(den) << ")";
                    }
                    return oss.str();
                };

                if (count == 1) {
                    value = read_srational(value_offset);
                } else {
                    json arr = json::array();
                    for (uint32_t i = 0; i < count; ++i) {
                        arr.push_back(read_srational(value_offset + i * 8));
                    }
                    value = std::move(arr);
                }
                break;
            }
            default:
                value = nullptr;
                break;
        }

        if (include_raw && raw_ptr != nullptr && raw_size != 0) {
            return json{
                {"decoded", value},
                {"raw_hex_preview", hex_preview(raw_ptr, raw_size)},
            };
        }

        return value;
    }

    json parse_ifd(const std::vector<uint8_t>& data,
                   size_t tiff_start,
                   uint32_t offset,
                   bool little_endian,
                   const std::string& ifd_name,
                   bool include_raw,
                   std::set<uint32_t>& visited,
                   std::string& warning) {
        if (offset == 0) {
            return nullptr;
        }
        if (!visited.insert(offset).second) {
            warning = "detected recursive IFD pointer";
            return nullptr;
        }
        if (tiff_start + offset + 2 > data.size()) {
            warning = "IFD offset out of range";
            return nullptr;
        }

        const size_t ifd_offset = tiff_start + offset;
        const uint16_t count    = read_u16_tiff(data, ifd_offset, little_endian);
        if (ifd_offset + 2 + static_cast<size_t>(count) * 12 + 4 > data.size()) {
            warning = "IFD entries exceed Exif payload size";
            return nullptr;
        }

        json ifd;
        ifd["name"] = ifd_name;
        ifd["tags"] = json::array();

        std::vector<std::pair<std::string, uint32_t>> subifds;

        for (uint16_t i = 0; i < count; ++i) {
            const size_t entry_offset    = ifd_offset + 2 + static_cast<size_t>(i) * 12;
            const uint16_t tag           = read_u16_tiff(data, entry_offset, little_endian);
            const uint16_t type          = read_u16_tiff(data, entry_offset + 2, little_endian);
            const uint32_t value_count   = read_u32_tiff(data, entry_offset + 4, little_endian);
            const uint32_t value_pointer = read_u32_tiff(data, entry_offset + 8, little_endian);

            json tag_json;
            tag_json["id"]    = tag;
            tag_json["name"]  = tag_name_for(tag, ifd_name);
            tag_json["type"]  = exif_type_name(type);
            tag_json["count"] = value_count;

            const size_t unit_size = exif_type_size(type);
            if (unit_size == 0) {
                tag_json["error"] = "unsupported Exif type";
                ifd["tags"].push_back(tag_json);
                continue;
            }

            const size_t total_size = unit_size * static_cast<size_t>(value_count);
            size_t value_offset     = 0;
            const uint8_t* raw_ptr  = nullptr;
            if (total_size <= 4) {
                value_offset = entry_offset + 8;
                raw_ptr      = data.data() + value_offset;
            } else {
                if (tiff_start + value_pointer + total_size > data.size()) {
                    tag_json["error"] = "Exif value points outside payload";
                    ifd["tags"].push_back(tag_json);
                    continue;
                }
                value_offset = tiff_start + value_pointer;
                raw_ptr      = data.data() + value_offset;
            }

            tag_json["value"] = parse_exif_value(data,
                                                 value_offset,
                                                 tag,
                                                 type,
                                                 value_count,
                                                 little_endian,
                                                 include_raw,
                                                 raw_ptr,
                                                 total_size);
            ifd["tags"].push_back(tag_json);

            if ((tag == 0x8769 && (ifd_name == "IFD0" || ifd_name == "IFD1"))) {
                subifds.push_back({"Exif", value_pointer});
            } else if ((tag == 0x8825 && (ifd_name == "IFD0" || ifd_name == "IFD1"))) {
                subifds.push_back({"GPS", value_pointer});
            } else if (tag == 0xA005 && ifd_name == "Exif") {
                subifds.push_back({"Interop", value_pointer});
            }
        }

        const uint32_t next_ifd_offset =
            read_u32_tiff(data, ifd_offset + 2 + static_cast<size_t>(count) * 12, little_endian);

        json children = json::array();
        for (const auto& [child_name, child_offset] : subifds) {
            json child = parse_ifd(data,
                                   tiff_start,
                                   child_offset,
                                   little_endian,
                                   child_name,
                                   include_raw,
                                   visited,
                                   warning);
            if (!child.is_null()) {
                children.push_back(child);
            }
        }
        if (!children.empty()) {
            ifd["children"] = std::move(children);
        }

        if (next_ifd_offset != 0 && (ifd_name == "IFD0" || ifd_name == "IFD1")) {
            json next_ifd = parse_ifd(data,
                                      tiff_start,
                                      next_ifd_offset,
                                      little_endian,
                                      "IFD1",
                                      include_raw,
                                      visited,
                                      warning);
            if (!next_ifd.is_null()) {
                ifd["next_ifd"] = std::move(next_ifd);
            }
        }

        return ifd;
    }

    json parse_exif_tiff(const uint8_t* payload,
                         size_t size,
                         bool include_raw,
                         std::string& error) {
        std::vector<uint8_t> data(payload, payload + size);
        json result;

        if (data.size() < 8) {
            error = "Exif payload too small";
            return result;
        }

        bool little_endian = false;
        if (data[0] == 'I' && data[1] == 'I') {
            little_endian = true;
        } else if (data[0] == 'M' && data[1] == 'M') {
            little_endian = false;
        } else {
            error = "invalid TIFF byte order";
            return result;
        }

        const uint16_t magic = read_u16_tiff(data, 2, little_endian);
        if (magic != 42) {
            std::ostringstream oss;
            oss << "unsupported TIFF magic " << magic;
            error = oss.str();
            return result;
        }

        std::set<uint32_t> visited;
        std::string warning;
        json ifd0 = parse_ifd(data,
                              0,
                              read_u32_tiff(data, 4, little_endian),
                              little_endian,
                              "IFD0",
                              include_raw,
                              visited,
                              warning);

        result["byte_order"] = little_endian ? "little_endian" : "big_endian";
        result["ifds"]       = json::array();
        if (!ifd0.is_null()) {
            result["ifds"].push_back(ifd0);
        }
        if (!warning.empty()) {
            result["warning"] = warning;
        }
        return result;
    }

    bool parse_png(const std::vector<uint8_t>& data,
                   bool include_raw,
                   json& result,
                   std::string& error) {
        static constexpr uint8_t sig[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
        if (data.size() < sizeof(sig) ||
            !std::equal(std::begin(sig), std::end(sig), data.begin())) {
            error = "not a PNG file";
            return false;
        }

        result["format"]  = "PNG";
        result["entries"] = json::array();

        size_t offset = sizeof(sig);
        while (offset + 12 <= data.size()) {
            const uint32_t length = read_u32_be(data, offset);
            const std::string type =
                bytes_to_string(data.data() + offset + 4, data.data() + offset + 8);
            offset += 8;

            if (offset + static_cast<size_t>(length) + 4 > data.size()) {
                error = "PNG chunk exceeds file size";
                return false;
            }

            const uint8_t* payload = data.data() + offset;
            const uint32_t crc     = read_u32_be(data, offset + length);

            json entry;
            entry["entry_type"] = "chunk";
            entry["name"]       = type;
            entry["length"]     = length;
            entry["crc"]        = crc;
            entry["critical"]   = !type.empty() && std::isupper(static_cast<unsigned char>(type[0])) != 0;
            entry["metadata_like"] =
                !(type == "IHDR" || type == "PLTE" || type == "IDAT" || type == "IEND");

            if (type == "IHDR" && length == 13) {
                entry["data"] = json{
                    {"width", read_u32_be(data, offset)},
                    {"height", read_u32_be(data, offset + 4)},
                    {"bit_depth", payload[8]},
                    {"color_type", payload[9]},
                    {"compression_method", payload[10]},
                    {"filter_method", payload[11]},
                    {"interlace_method", payload[12]},
                };
            } else if (type == "tEXt") {
                const uint8_t* separator =
                    static_cast<const uint8_t*>(memchr(payload, '\0', length));
                if (separator != nullptr) {
                    entry["data"] = json{
                        {"keyword", bytes_to_string(payload, separator)},
                        {"text", bytes_to_string(separator + 1, payload + length)},
                    };
                }
            } else if (type == "zTXt") {
                const uint8_t* separator =
                    static_cast<const uint8_t*>(memchr(payload, '\0', length));
                if (separator != nullptr && separator + 2 <= payload + length) {
                    json meta;
                    meta["keyword"]            = bytes_to_string(payload, separator);
                    meta["compression_method"] = separator[1];
                    std::string text;
                    std::string decompress_error;
                    if (decompress_zlib(separator + 2,
                                        static_cast<size_t>(payload + length - (separator + 2)),
                                        text,
                                        decompress_error)) {
                        meta["text"] = text;
                    } else {
                        meta["decompression_error"] = decompress_error;
                    }
                    entry["data"] = std::move(meta);
                }
            } else if (type == "iTXt") {
                const uint8_t* keyword_end =
                    static_cast<const uint8_t*>(memchr(payload, '\0', length));
                if (keyword_end != nullptr && keyword_end + 3 <= payload + length) {
                    json meta;
                    meta["keyword"] = bytes_to_string(payload, keyword_end);

                    const uint8_t compression_flag   = keyword_end[1];
                    const uint8_t compression_method = keyword_end[2];
                    const uint8_t* cursor            = keyword_end + 3;

                    const uint8_t* language_end =
                        static_cast<const uint8_t*>(memchr(cursor, '\0', payload + length - cursor));
                    if (language_end == nullptr) {
                        language_end = payload + length;
                    }
                    meta["language_tag"] = bytes_to_string(cursor, language_end);
                    cursor               = std::min(language_end + 1, payload + length);

                    const uint8_t* translated_end =
                        static_cast<const uint8_t*>(memchr(cursor, '\0', payload + length - cursor));
                    if (translated_end == nullptr) {
                        translated_end = payload + length;
                    }
                    meta["translated_keyword"] = bytes_to_string(cursor, translated_end);
                    cursor                     = std::min(translated_end + 1, payload + length);

                    meta["compression_flag"]   = compression_flag;
                    meta["compression_method"] = compression_method;

                    std::string text;
                    std::string decompress_error;
                    if (compression_flag == 1) {
                        if (decompress_zlib(cursor,
                                            static_cast<size_t>(payload + length - cursor),
                                            text,
                                            decompress_error)) {
                            meta["text"] = text;
                        } else {
                            meta["decompression_error"] = decompress_error;
                        }
                    } else {
                        meta["text"] = bytes_to_string(cursor, payload + length);
                    }
                    entry["data"] = std::move(meta);
                }
            } else if (type == "eXIf") {
                std::string exif_error;
                json meta = parse_exif_tiff(payload, length, include_raw, exif_error);
                if (!meta.empty()) {
                    entry["data"] = std::move(meta);
                }
                if (!exif_error.empty()) {
                    entry["error"] = exif_error;
                }
            } else if (type == "iCCP") {
                const uint8_t* separator =
                    static_cast<const uint8_t*>(memchr(payload, '\0', length));
                if (separator != nullptr && separator + 2 <= payload + length) {
                    json meta;
                    meta["profile_name"]       = bytes_to_string(payload, separator);
                    meta["compression_method"] = separator[1];
                    std::string profile;
                    std::string decompress_error;
                    if (decompress_zlib(separator + 2,
                                        static_cast<size_t>(payload + length - (separator + 2)),
                                        profile,
                                        decompress_error)) {
                        meta["profile_size"] = profile.size();
                        if (include_raw) {
                            meta["profile_hex_preview"] =
                                hex_preview(reinterpret_cast<const uint8_t*>(profile.data()), profile.size());
                        }
                    } else {
                        meta["decompression_error"] = decompress_error;
                    }
                    entry["data"] = std::move(meta);
                }
            } else if (type == "gAMA" && length == 4) {
                entry["data"] = json{{"gamma_times_100000", read_u32_be(data, offset)}};
            } else if (type == "cHRM" && length == 32) {
                entry["data"] = json{
                    {"white_point_x", read_u32_be(data, offset)},
                    {"white_point_y", read_u32_be(data, offset + 4)},
                    {"red_x", read_u32_be(data, offset + 8)},
                    {"red_y", read_u32_be(data, offset + 12)},
                    {"green_x", read_u32_be(data, offset + 16)},
                    {"green_y", read_u32_be(data, offset + 20)},
                    {"blue_x", read_u32_be(data, offset + 24)},
                    {"blue_y", read_u32_be(data, offset + 28)},
                };
            } else if (type == "sRGB" && length == 1) {
                entry["data"] = json{{"rendering_intent", payload[0]}};
            } else if (type == "pHYs" && length == 9) {
                entry["data"] = json{
                    {"pixels_per_unit_x", read_u32_be(data, offset)},
                    {"pixels_per_unit_y", read_u32_be(data, offset + 4)},
                    {"unit_specifier", payload[8]},
                };
            } else if (type == "tIME" && length == 7) {
                entry["data"] = json{
                    {"year", read_u16_be(data, offset)},
                    {"month", payload[2]},
                    {"day", payload[3]},
                    {"hour", payload[4]},
                    {"minute", payload[5]},
                    {"second", payload[6]},
                };
            } else {
                append_raw_preview(entry, payload, length, include_raw);
            }

            result["entries"].push_back(entry);
            offset += static_cast<size_t>(length) + 4;

            if (type == "IEND") {
                return true;
            }
        }

        error = "PNG missing IEND chunk";
        return false;
    }

    bool parse_jpeg(const std::vector<uint8_t>& data,
                    bool include_raw,
                    json& result,
                    std::string& error) {
        if (data.size() < 2 || data[0] != 0xFF || data[1] != 0xD8) {
            error = "not a JPEG file";
            return false;
        }

        result["format"]  = "JPEG";
        result["entries"] = json::array();

        size_t offset = 2;
        while (offset + 1 < data.size()) {
            if (data[offset] != 0xFF) {
                error = "invalid JPEG marker alignment";
                return false;
            }

            while (offset < data.size() && data[offset] == 0xFF) {
                ++offset;
            }
            if (offset >= data.size()) {
                break;
            }

            const uint8_t marker = data[offset++];
            if (marker == 0xD9 || marker == 0xDA) {
                break;
            }
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                continue;
            }
            if (offset + 2 > data.size()) {
                error = "JPEG marker missing segment length";
                return false;
            }

            const uint16_t segment_length = read_u16_be(data, offset);
            if (segment_length < 2 || offset + segment_length > data.size()) {
                error = "JPEG segment exceeds file size";
                return false;
            }
            offset += 2;

            const uint8_t* payload    = data.data() + offset;
            const size_t payload_size = segment_length - 2;
            json entry;
            entry["entry_type"] = "segment";
            entry["marker"]     = marker_name(marker);
            entry["length"]     = payload_size;
            entry["metadata_like"] =
                (marker == 0xFE) || (marker >= 0xE0 && marker <= 0xEF);

            if (marker == 0xFE) {
                std::string comment = bytes_to_string(payload, payload + payload_size);
                if (comment.rfind("parameters", 0) == 0 &&
                    comment.size() > std::string("parameters").size() &&
                    comment[std::string("parameters").size()] == '\0') {
                    entry["data"] = json{
                        {"label", "parameters"},
                        {"text", comment.substr(std::string("parameters").size() + 1)},
                    };
                } else {
                    entry["data"] = json{{"text", trim_trailing_nuls(comment)}};
                }
            } else if (marker == 0xE0 && payload_size >= 5 &&
                       memcmp(payload, "JFIF\0", 5) == 0) {
                entry["data"] = json{
                    {"identifier", "JFIF"},
                    {"version_major", payload_size >= 7 ? payload[5] : 0},
                    {"version_minor", payload_size >= 7 ? payload[6] : 0},
                    {"density_units", payload_size >= 8 ? payload[7] : 0},
                    {"x_density", payload_size >= 10 ? read_u16_be(data, offset + 8) : 0},
                    {"y_density", payload_size >= 12 ? read_u16_be(data, offset + 10) : 0},
                };
            } else if (marker == 0xE1 && payload_size >= 6 &&
                       memcmp(payload, "Exif\0\0", 6) == 0) {
                std::string exif_error;
                json meta = parse_exif_tiff(payload + 6, payload_size - 6, include_raw, exif_error);
                if (!meta.empty()) {
                    entry["data"] = std::move(meta);
                }
                if (!exif_error.empty()) {
                    entry["error"] = exif_error;
                }
            } else if (marker == 0xE1 && payload_size >= 29 &&
                       memcmp(payload, "http://ns.adobe.com/xap/1.0/\0", 29) == 0) {
                entry["data"] = json{
                    {"type", "XMP"},
                    {"xml", bytes_to_string(payload + 29, payload + payload_size)},
                };
            } else if (marker == 0xE2 && payload_size >= 14 &&
                       memcmp(payload, "ICC_PROFILE\0", 12) == 0) {
                json meta;
                meta["type"]           = "ICC_PROFILE";
                meta["sequence_no"]    = payload[12];
                meta["sequence_count"] = payload[13];
                meta["profile_size"]   = payload_size - 14;
                if (include_raw && payload_size > 14) {
                    meta["profile_hex_preview"] = hex_preview(payload + 14, payload_size - 14);
                }
                entry["data"] = std::move(meta);
            } else if (marker == 0xEE && payload_size >= 12 &&
                       memcmp(payload, "Adobe\0", 6) == 0) {
                entry["data"] = json{
                    {"identifier", "Adobe"},
                    {"version", read_u16_be(data, offset + 6)},
                    {"flags0", read_u16_be(data, offset + 8)},
                    {"flags1", read_u16_be(data, offset + 10)},
                    {"color_transform", payload[11]},
                };
            } else {
                append_raw_preview(entry, payload, payload_size, include_raw);
            }

            result["entries"].push_back(entry);
            offset += payload_size;
        }

        return true;
    }

    bool parse_webp(const std::vector<uint8_t>& data,
                    bool include_raw,
                    json& result,
                    std::string& error) {
        if (data.size() < 12 ||
            memcmp(data.data(), "RIFF", 4) != 0 ||
            memcmp(data.data() + 8, "WEBP", 4) != 0) {
            error = "not a WebP file";
            return false;
        }

        result["format"]  = "WEBP";
        result["entries"] = json::array();

        size_t offset = 12;
        while (offset + 8 <= data.size()) {
            const std::string raw_type =
                bytes_to_string(data.data() + offset, data.data() + offset + 4);
            const uint32_t length = read_u32_le(data, offset + 4);
            offset += 8;

            if (offset + static_cast<size_t>(length) > data.size()) {
                error = "WebP chunk exceeds file size";
                return false;
            }

            const uint8_t* payload = data.data() + offset;
            const std::string type =
                !raw_type.empty() && raw_type.back() == ' '
                    ? raw_type.substr(0, raw_type.size() - 1)
                    : raw_type;

            json entry;
            entry["entry_type"] = "chunk";
            entry["name"]       = type;
            entry["length"]     = length;
            entry["metadata_like"] =
                (raw_type == "ICCP" || raw_type == "EXIF" || raw_type == "XMP ");

            if (raw_type == "VP8X" && length >= 10) {
                entry["data"] = json{
                    {"icc_profile", (payload[0] & 0x20) != 0},
                    {"alpha", (payload[0] & 0x10) != 0},
                    {"exif", (payload[0] & 0x08) != 0},
                    {"xmp", (payload[0] & 0x04) != 0},
                    {"animation", (payload[0] & 0x02) != 0},
                    {"canvas_width", 1 + static_cast<uint32_t>(payload[4]) + (static_cast<uint32_t>(payload[5]) << 8) + (static_cast<uint32_t>(payload[6]) << 16)},
                    {"canvas_height", 1 + static_cast<uint32_t>(payload[7]) + (static_cast<uint32_t>(payload[8]) << 8) + (static_cast<uint32_t>(payload[9]) << 16)},
                };
            } else if (raw_type == "EXIF") {
                std::string exif_error;
                json meta = parse_exif_tiff(payload, length, include_raw, exif_error);
                if (!meta.empty()) {
                    entry["data"] = std::move(meta);
                }
                if (!exif_error.empty()) {
                    entry["error"] = exif_error;
                }
            } else if (raw_type == "XMP ") {
                entry["data"] = json{
                    {"type", "XMP"},
                    {"xml", trim_trailing_nuls(bytes_to_string(payload, payload + length))},
                };
            } else if (raw_type == "ICCP") {
                entry["data"] = json{{"profile_size", length}};
                append_raw_preview(entry["data"], payload, length, include_raw);
            } else {
                append_raw_preview(entry, payload, length, include_raw);
            }

            result["entries"].push_back(entry);
            offset += static_cast<size_t>(length) + (length & 1u);
        }

        return true;
    }

    static bool parse_avi_chunks(const std::vector<uint8_t>& data, size_t offset, size_t end, bool info, bool include_raw, json& result, std::string& error) {
        while (offset < end) {
            if (end - offset < 8) {
                error = "truncated AVI chunk header";
                return false;
            }
            const auto name   = bytes_to_string(data.data() + offset, data.data() + offset + 4);
            const size_t size = read_u32_le(data, offset + 4);
            offset += 8;
            if (size > end - offset || (size & 1u) > end - offset - size) {
                error = "AVI chunk exceeds its container";
                return false;
            }
            json entry = {{"entry_type", "chunk"}, {"name", name}, {"length", size}, {"metadata_like", info}};
            if (name == "LIST") {
                if (size < 4) {
                    error = "truncated AVI LIST type";
                    return false;
                }
                const auto type            = bytes_to_string(data.data() + offset, data.data() + offset + 4);
                entry["data"]["list_type"] = type;
                if (type == "INFO" && !info &&
                    !parse_avi_chunks(data, offset + 4, offset + size, true, include_raw, result, error))
                    return false;
            } else if (info) {
                entry["data"]["text"] = trim_trailing_nuls(bytes_to_string(data.data() + offset, data.data() + offset + size));
            } else {
                append_raw_preview(entry, data.data() + offset, size, include_raw);
            }
            result["entries"].push_back(std::move(entry));
            offset += size + (size & 1u);
        }
        return true;
    }

    static bool parse_avi(const std::vector<uint8_t>& data, bool include_raw, json& result, std::string& error) {
        result        = {{"format", "AVI"}, {"entries", json::array()}};
        size_t offset = 0;
        while (offset < data.size()) {
            if (data.size() - offset < 12 || memcmp(data.data() + offset, "RIFF", 4) != 0 ||
                memcmp(data.data() + offset + 8, offset == 0 ? "AVI " : "AVIX", 4) != 0) {
                error = "invalid AVI RIFF header";
                return false;
            }
            const size_t size = read_u32_le(data, offset + 4);
            if (size < 4 || size > data.size() - offset - 8 ||
                (size & 1u) > data.size() - offset - 8 - size) {
                error = "AVI RIFF size exceeds file size";
                return false;
            }
            if (!parse_avi_chunks(data, offset + 12, offset + 8 + size, false, include_raw, result, error))
                return false;
            offset += 8 + size + (size & 1u);
        }
        return true;
    }

    static bool read_ebml_uint(const std::vector<uint8_t>& data, size_t& offset, size_t end, bool id, uint64_t& value, bool& unknown) {
        if (offset == end || data[offset] == 0)
            return false;
        uint8_t mask  = 0x80;
        size_t length = 1;
        while (!(data[offset] & mask)) {
            mask >>= 1;
            ++length;
        }
        if (length > (id ? 4u : 8u) || length > end - offset)
            return false;
        value = data[offset++] & (id ? 0xff : mask - 1);
        for (size_t i = 1; i < length; ++i)
            value = (value << 8) | data[offset++];
        unknown = !id && value == (uint64_t(1) << (7 * length)) - 1;
        return true;
    }

    static bool is_webm_segment_child(uint64_t id) {
        return id == 0x114D9B74 || id == 0x1549A966 || id == 0x1654AE6B ||
               id == 0x1F43B675 || id == 0x1C53BB6B || id == 0x1941A469 ||
               id == 0x1043A770 || id == 0x1254C367;
    }

    static bool parse_webm_elements(const std::vector<uint8_t>& data, size_t& offset, size_t end, uint64_t parent, bool unknown_cluster, int depth, bool include_raw, json& result, std::string& error) {
        if (depth > 16) {
            error = "WebM tag nesting exceeds limit";
            return false;
        }
        std::string tag_name;
        std::string tag_text;
        bool has_text = false;
        while (offset < end) {
            const auto start = offset;
            uint64_t id, size;
            bool unknown;
            if (!read_ebml_uint(data, offset, end, true, id, unknown)) {
                error = "invalid or truncated WebM element ID";
                return false;
            }
            // An unknown-size Cluster ends at the next Segment-level element.
            if (unknown_cluster && is_webm_segment_child(id)) {
                offset = start;
                return true;
            }
            if (!read_ebml_uint(data, offset, end, false, size, unknown) ||
                (!unknown && size > end - offset)) {
                error = "invalid or truncated WebM element size";
                return false;
            }
            if (unknown && id != 0x18538067 && id != 0x1F43B675) {
                error = "unsupported unknown-size WebM element";
                return false;
            }
            const size_t stop    = unknown ? end : offset + static_cast<size_t>(size);
            const bool container = (parent == 0 && (id == 0x1A45DFA3 || id == 0x18538067)) ||
                                   (parent == 0x18538067 && (id == 0x1254C367 || (id == 0x1F43B675 && unknown))) ||
                                   (parent == 0x1254C367 && id == 0x7373) ||
                                   ((parent == 0x7373 || parent == 0x67C8) && id == 0x67C8);
            if (parent == 0 && id == 0x18538067)
                result["has_segment"] = true;
            if (parent == 0x1A45DFA3 && id == 0x4282) {
                const auto type = bytes_to_string(data.data() + offset, data.data() + stop);
                if (type != "webm" && type != "matroska") {
                    error = "EBML document is not WebM or Matroska";
                    return false;
                }
                result["doc_type"] = type;
                result["format"]   = type == "webm" ? "WEBM" : "MATROSKA";
            }
            if (parent == 0x67C8 && (id == 0x45A3 || id == 0x4487)) {
                auto value = bytes_to_string(data.data() + offset, data.data() + stop);
                if (id == 0x45A3)
                    tag_name = std::move(value);
                else {
                    tag_text = std::move(value);
                    has_text = true;
                }
            }
            if (parent == 0 || parent == 0x18538067) {
                std::ostringstream name;
                name << "0x" << std::hex << std::uppercase << id;
                json entry = {{"entry_type", "element"}, {"name", name.str()}, {"length", stop - offset}, {"metadata_like", false}};
                if (unknown)
                    entry["unknown_size"] = true;
                append_raw_preview(entry, data.data() + offset, stop - offset, include_raw);
                result["entries"].push_back(std::move(entry));
            }
            if (container) {
                if (!parse_webm_elements(data, offset, stop, id, id == 0x1F43B675 && unknown,
                                         depth + 1, include_raw, result, error))
                    return false;
            } else {
                offset = stop;
            }
        }
        if (parent == 0x67C8 && has_text) {
            result["entries"].push_back({{"entry_type", "tag"}, {"name", tag_name}, {"metadata_like", true}, {"data", {{"text", tag_text}}}});
        }
        return true;
    }

    static bool parse_webm(const std::vector<uint8_t>& data, bool include_raw, json& result, std::string& error) {
        result        = {{"format", "WEBM"}, {"entries", json::array()}};
        size_t offset = 0;
        if (!parse_webm_elements(data, offset, data.size(), 0, false, 0, include_raw, result, error))
            return false;
        if (!result.contains("doc_type") || !result.value("has_segment", false)) {
            error = "missing WebM document type or Segment";
            return false;
        }
        result.erase("has_segment");
        return true;
    }

    std::string abbreviate(const std::string& value, bool brief) {
        if (!brief || value.size() <= 240) {
            return value;
        }
        return value.substr(0, 240) + "...";
    }

    void print_json_value(std::ostream& out,
                          const std::string& key,
                          const json& value,
                          int indent,
                          bool brief) {
        const std::string prefix(indent, ' ');
        if (value.is_string()) {
            out << prefix << key << ": " << abbreviate(value.get<std::string>(), brief) << "\n";
            return;
        }
        if (value.is_primitive()) {
            out << prefix << key << ": " << value.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
            return;
        }
        if (value.is_array()) {
            out << prefix << key << ":\n";
            for (const auto& item : value) {
                if (item.is_string()) {
                    out << prefix << "  - " << abbreviate(item.get<std::string>(), brief) << "\n";
                } else if (item.is_primitive()) {
                    out << prefix << "  - " << item.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
                } else {
                    out << prefix << "  -\n";
                    for (auto it = item.begin(); it != item.end(); ++it) {
                        print_json_value(out, it.key(), it.value(), indent + 4, brief);
                    }
                }
            }
            return;
        }

        out << prefix << key << ":\n";
        for (auto it = value.begin(); it != value.end(); ++it) {
            print_json_value(out, it.key(), it.value(), indent + 2, brief);
        }
    }

    void print_text_report(const std::string& path,
                           const json& report,
                           bool include_structural,
                           bool brief,
                           std::ostream& out) {
        (void)include_structural;
        out << "File: " << path << "\n";
        out << "Format: " << report.value("format", "unknown") << "\n\n";

        const auto& entries = report["entries"];
        if (entries.empty()) {
            out << "No metadata entries found.\n";
            return;
        }

        for (const auto& entry : entries) {
            const auto type   = entry.value("entry_type", "");
            const auto name   = entry.value("name", entry.value("marker", "unknown"));
            const char* label = type == "chunk" ? "Chunk" : type == "tag"   ? "Tag"
                                                        : type == "element" ? "Element"
                                                                            : "Segment";
            out << label << ": " << name << "\n";
            for (auto it = entry.begin(); it != entry.end(); ++it) {
                if (it.key() == "entry_type" || it.key() == "name" || it.key() == "marker" ||
                    it.key() == "metadata_like") {
                    continue;
                }
                print_json_value(out, it.key(), it.value(), 2, brief);
            }
            out << "\n";
        }
    }

    json filter_visible_entries(const json& report, bool include_structural) {
        json filtered = report;
        if (!filtered.contains("entries") || !filtered["entries"].is_array()) {
            return filtered;
        }

        filtered["entries"] = json::array();
        for (const auto& entry : report["entries"]) {
            if (!include_structural && !entry.value("metadata_like", false)) {
                continue;
            }
            json visible_entry = entry;
            visible_entry.erase("metadata_like");
            filtered["entries"].push_back(std::move(visible_entry));
        }
        return filtered;
    }

    bool build_metadata_report(const std::string& image_path,
                               bool include_raw,
                               json& report,
                               std::string& error) {
        std::vector<uint8_t> data;
        if (!read_file(image_path, data, error)) {
            return false;
        }
        if (data.size() >= 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' &&
            data[3] == 'G') {
            return parse_png(data, include_raw, report, error);
        }
        if (data.size() >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
            return parse_jpeg(data, include_raw, report, error);
        }
        if (data.size() >= 12 && memcmp(data.data(), "RIFF", 4) == 0 &&
            memcmp(data.data() + 8, "WEBP", 4) == 0) {
            return parse_webp(data, include_raw, report, error);
        }

        if (data.size() >= 12 && memcmp(data.data(), "RIFF", 4) == 0 &&
            memcmp(data.data() + 8, "AVI ", 4) == 0)
            return parse_avi(data, include_raw, report, error);
        if (data.size() >= 4 && read_u32_be(data, 0) == 0x1A45DFA3)
            return parse_webm(data, include_raw, report, error);

        error = "unsupported media format; only PNG, JPEG, WebP, AVI, WebM, and Matroska are supported";
        return false;
    }

}  // namespace

bool print_image_metadata(const std::string& image_path,
                          const MetadataReadOptions& options,
                          std::ostream& out,
                          std::string& error) {
    json report;
    if (!build_metadata_report(image_path, options.include_raw, report, error)) {
        return false;
    }

    json visible_report = filter_visible_entries(report, options.include_structural);

    if (options.output_format == MetadataOutputFormat::JSON) {
        visible_report["file"] = image_path;
        out << visible_report.dump(2, ' ', false, json::error_handler_t::replace) << "\n";
    } else {
        print_text_report(image_path, visible_report, options.include_structural, options.brief, out);
    }
    return true;
}
